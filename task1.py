from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import google.generativeai as genai
import os
import tempfile
import fitz
import docx
from typing import List
from pydantic import BaseModel
import uvicorn
import mimetypes
from dotenv import load_dotenv
from collections import Counter
import re

# Load environment variables
load_dotenv()

# Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="Job Criteria Extractor API")

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

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Job Criteria Extractor API Documentation"
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
