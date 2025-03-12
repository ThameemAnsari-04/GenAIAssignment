from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
import google.generativeai as genai
import os
import tempfile
import fitz
import docx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
import mimetypes
from dotenv import load_dotenv
from collections import Counter
import re
import pandas as pd
import json
from io import BytesIO
import uuid
from openpyxl.workbook import Workbook

# Load environment variables
load_dotenv()

# Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="Job Criteria Extractor and Resume Scorer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CriteriaResponse(BaseModel):
    criteria: List[str]

# keywords ranking
KEYWORD_WEIGHTS = {
    'required': 2,
    'must-have': 3,
    'essential': 2,
    'preferred': 1,
    'desired': 1,
    'mandatory': 3
}

def calculate_keyword_weight(text: str) -> int:
    """
    Calculate a weight score based on keyword presence in the job description.
    
    - **text**: The job description text.
    Returns a weight score based on the frequency of the keyword matches.
    """
    weight_score = 0
    normalized_text = text.lower()

    for keyword, weight in KEYWORD_WEIGHTS.items():
        keyword_count = normalized_text.count(keyword.lower())
        weight_score += keyword_count * weight
    
    return weight_score

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from DOCX: {str(e)}")

def rank_criteria_with_keywords(criteria: List[str], text: str) -> List[str]:
    """
    Rank the criteria based on their frequency and keyword weights in the job description text.
    
    - **criteria**: List of extracted criteria
    - **text**: Job description text
    
    Returns a list of criteria ranked from most to least important.
    """
    normalized_text = re.sub(r'\W+', ' ', text.lower())
    
    criteria_freq = Counter()
    for criterion in criteria:
        criterion_normalized = re.sub(r'\W+', ' ', criterion.lower())
        criteria_freq[criterion] = normalized_text.count(criterion_normalized)
    
    weighted_criteria_scores = []
    for criterion in criteria:
        freq_score = criteria_freq[criterion.lower()]
        keyword_weight = calculate_keyword_weight(text)
        total_score = freq_score + keyword_weight
        weighted_criteria_scores.append((criterion, total_score))
    
    ranked_criteria = [criterion for criterion, _ in sorted(weighted_criteria_scores, key=lambda x: x[1], reverse=True)]
    
    return ranked_criteria

def extract_criteria_with_genai(text):
    """Use Google Generative AI to extract ranking criteria from text."""
    prompt = f"""
    From the following job description, extract all ranking criteria including required skills, 
    certifications, experience, qualifications, and other important requirements. 
    Return ONLY a list of criteria, each as a separate string. Each criterion should be specific 
    and clearly defined.

    Job Description:
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Process the response to get a clean list of criteria
        response_text = response.text
        criteria_list = [
            line.strip() for line in response_text.split('\n') 
            if line.strip() and not line.strip().startswith('- ')
        ]
        
        if not criteria_list and '- ' in response_text:
            criteria_list = [
                line.strip()[2:] for line in response_text.split('\n') 
                if line.strip().startswith('- ')
            ]
        
        # Rank the criteria based on frequency and keyword weighting
        ranked_criteria = rank_criteria_with_keywords(criteria_list, text)
        
        return ranked_criteria
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error using Google Generative AI: {str(e)}")

@app.post("/extract-criteria", response_model=CriteriaResponse, tags=["Extraction"])
async def extract_criteria(file: UploadFile = File(...)):
    """
    Extract ranking criteria from a job description file (PDF or DOCX).
    
    - **file**: Upload a PDF or DOCX file containing the job description
    
    Returns a JSON object with a list of ranked criteria.
    """
    # Checking file type
    content_type = file.content_type
    if not content_type:
        content_type, _ = mimetypes.guess_type(file.filename)
    
    if not content_type or not (
        content_type == "application/pdf" or 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or
        content_type == "application/msword"
    ):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are supported")
    
    # Saving the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        file_content = await file.read()
        temp_file.write(file_content)
    
    try:
        # Extracting text based on file type
        if content_type == "application/pdf":
            text = extract_text_from_pdf(temp_file_path)
        else: 
            text = extract_text_from_docx(temp_file_path)
        
        # Extracting and ranking criteria using Gemini with keyword ranking
        ranked_criteria = extract_criteria_with_genai(text)
        
        return CriteriaResponse(criteria=ranked_criteria)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def extract_candidate_name(resume_text: str) -> str:
    """Extract candidate name from resume text using AI."""
    prompt = f"""
    Extract the candidate's full name from the following resume text. 
    Return ONLY the name, with no additional text or explanation.
    
    Resume:
    {resume_text[:2000]}  # Using first 2000 chars to stay within token limits
    """
    
    try:
        response = model.generate_content(prompt)
        name = response.text.strip()
        
        # If the name is too long, it's probably not just a name
        if len(name.split()) > 5:
            # Fallback to a simple heuristic - first line often contains the name
            first_lines = resume_text.strip().split('\n')[:5]
            for line in first_lines:
                if len(line.strip()) > 0 and len(line.strip().split()) <= 5:
                    return line.strip()
            return "Unknown Candidate"
        
        return name
    except Exception:
        # Fallback to a simple heuristic
        first_line = resume_text.strip().split('\n')[0]
        if len(first_line.strip()) > 0 and len(first_line.strip().split()) <= 5:
            return first_line.strip()
        return "Unknown Candidate"

def score_resume_for_criterion(resume_text: str, criterion: str) -> int:
    """
    Score a resume against a specific criterion using AI.
    Returns a score from 0-5.
    """
    prompt = f"""
    Evaluate how well the candidate's resume matches the following criterion:
    
    Criterion: {criterion}
    
    Resume:
    {resume_text[:3000]}  # Using first 3000 chars to stay within token limits
    
    Score the match on a scale of 0-5, where:
    0 = No match/Not mentioned
    1 = Barely mentioned, no relevant experience
    2 = Mentioned but limited experience
    3 = Moderate match with some relevant experience
    4 = Good match with relevant experience
    5 = Excellent match with extensive experience
    
    Return ONLY the numeric score (0, 1, 2, 3, 4, or 5) with no additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
        # Extract just the numeric score
        score_match = re.search(r'[0-5]', score_text)
        if score_match:
            return int(score_match.group(0))
        else:
            # If no clear score is found, make a conservative estimate
            return 2
    except Exception as e:
        # If there's an error, default to a neutral score
        print(f"Error scoring criterion '{criterion}': {str(e)}")
        return 2

async def parse_criteria_string(criteria_str: str) -> List[str]:
    """Parse criteria string into a list."""
    try:
        # Try to parse as JSON
        criteria_list = json.loads(criteria_str)
        if isinstance(criteria_list, list):
            return criteria_list
    except:
        pass
    
    # If not JSON, try to split by newlines or commas
    if '\n' in criteria_str:
        return [c.strip() for c in criteria_str.split('\n') if c.strip()]
    else:
        return [c.strip() for c in criteria_str.split(',') if c.strip()]

@app.post("/score-resumes", tags=["Scoring"])
async def score_resumes(
    criteria: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Score multiple resumes against provided criteria and return results as Excel/CSV.
    
    - **criteria**: List of criteria as a JSON string or comma-separated values
    - **files**: Multiple resume files (PDF or DOCX)
    
    Returns an Excel file with scores for each candidate against each criterion.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No resume files provided")
    
    # Parse criteria
    criteria_list = await parse_criteria_string(criteria)
    if not criteria_list:
        raise HTTPException(status_code=400, detail="No valid criteria provided")
    
    # Create a list to store results for each candidate
    results = []
    
    # Process each resume
    for file in files:
        # Check file type
        content_type = file.content_type
        if not content_type:
            content_type, _ = mimetypes.guess_type(file.filename)
        
        if not content_type or not (
            content_type == "application/pdf" or 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type or
            content_type == "application/msword"
        ):
            continue  # Skip unsupported files
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            file_content = await file.read()
            temp_file.write(file_content)
        
        try:
            # Extract text from the resume
            resume_text = extract_text_from_file(temp_file_path)
            
            # Extract candidate name
            candidate_name = extract_candidate_name(resume_text)
            
            # Score the resume against each criterion
            scores = {}
            total_score = 0
            
            for criterion in criteria_list:
                score = score_resume_for_criterion(resume_text, criterion)
                
                # Store the criterion name and score
                criterion_key = criterion[:30]  # Truncate long criterion names for column headers
                scores[criterion_key] = score
                total_score += score
            
            # Add total score
            scores["Total Score"] = total_score
            
            # Add to results
            result = {"Candidate Name": candidate_name}
            result.update(scores)
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            continue
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    if not results:
        raise HTTPException(status_code=400, detail="No valid resumes could be processed")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Create Excel file
    output_file = f"resume_scores_{uuid.uuid4()}.xlsx"
    df.to_excel(output_file, index=False)
    
    # Return the Excel file
    response = FileResponse(
        path=output_file, 
        filename="resume_scores.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    return response

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Job Criteria Extractor and Resume Scorer API Documentation"
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)