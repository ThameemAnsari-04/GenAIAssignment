Resume Criteria Extractor and Scorer
This application is a powerful tool for HR professionals and recruiters to streamline the candidate evaluation process. It provides two main functionalities:

Job Criteria Extraction: Automatically extract key requirements and criteria from job descriptions.

Resume Scoring: Score multiple resumes against extracted criteria to identify the best matches.

Overview
The system uses FastAPI to provide a robust API interface and leverages Google's Generative AI (Gemini 2.0) to perform intelligent text analysis. It can process PDF and DOCX files, extract meaningful information, and generate structured outputs for decision-making.

Features
1. Job Criteria Extraction
Upload job descriptions as PDF or DOCX files

Automatically extract key requirements, skills, and qualifications

Rank criteria by importance using keyword analysis

Return a structured list of criteria for evaluation

2. Resume Scoring
Upload multiple resumes (PDF or DOCX)

Score each resume against provided criteria

Extract candidate names automatically

Generate an Excel report with individual and total scores

Rank candidates based on their match to job requirements

Technical Implementation
The application is built with:

FastAPI: For creating robust, high-performance API endpoints

Google Gemini 2.0: For AI-powered text analysis and scoring

PyMuPDF & python-docx: For document text extraction

Pandas: For data processing and Excel report generation

Setup and Installation
Prerequisites
Python 3.8 or higher

Google Generative AI API key

Environment Setup
Clone the repository:
git clone https://github.com/yourusername/resume-criteria-extractor.git
cd resume-criteria-extractor

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Create a .env file with your Google API key:
GOOGLE_API_KEY=your_api_key_here

Running the Application
Start the FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
The API will be accessible at http://localhost:8000, and the interactive documentation at http://localhost:8000/docs.

API Usage
Extract Criteria from Job Description
Endpoint: POST /extract-criteria

Input:

A job description file (PDF or DOCX)

Output:

A JSON list of ranked criteria extracted from the job description

Example using curl:

curl -X POST "http://localhost:8000/extract-criteria" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@job_description.pdf"
Score Resumes Against Criteria
Endpoint: POST /score-resumes

Input:

A list of criteria (as a JSON string or comma-separated values)

Multiple resume files (PDF or DOCX)

Output:

An Excel file with scores for each candidate against each criterion

Example using curl:

curl -X POST "http://localhost:8000/score-resumes" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "criteria=[\"5+ years of Python experience\", \"Machine Learning expertise\", \"AWS certification\"]" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx" \
  -F "files=@resume3.pdf" \
  --output resume_scores.xlsx
How It Works
Job Criteria Extraction Process
The user uploads a job description document

The system extracts text from the document based on its format

The text is sent to Google's Gemini AI with a specialized prompt

The AI identifies key requirements, skills, and qualifications

The system ranks the criteria based on keyword importance

A structured list of criteria is returned to the user

Resume Scoring Process
The user provides a list of criteria and uploads multiple resumes

For each resume:
Text is extracted from the document
The candidate's name is identified
Each criterion is evaluated against the resume content
A score from 0-5 is assigned for each criterion
A total score is calculated

All scores are compiled into an Excel spreadsheet

The spreadsheet is returned to the user for analysis

Scoring Methodology
Resumes are scored on a scale of 0-5 for each criterion:

0: No match/Not mentioned

1: Barely mentioned, no relevant experience

2: Mentioned but limited experience

3: Moderate match with some relevant experience

4: Good match with relevant experience

5: Excellent match with extensive experience

The AI analyzes the context and content of the resume to determine how well the candidate's experience matches each requirement.

Dependencies
fastapi

uvicorn

python-multipart

google-generativeai

pymupdf

python-docx

pandas

openpyxl

python-dotenv

Future Enhancements
Support for more document formats

Enhanced candidate information extraction

Custom scoring weights for different criteria

Interactive web interface

Batch processing of large numbers of resumes

Integration with ATS (Applicant Tracking Systems)

License
This project is licensed under the MIT License - see the LICENSE file for details.
