from crewai import Agent, Task, Crew, LLM
import os
import PyPDF2
from pydantic import BaseModel
from typing import List
import re
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
import streamlit as st
from tools import *


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

# Define schemas for each field
uid_schema = ResponseSchema(
    name="UID",
    description=(
        "A unique identifier for the candidate, constructed using the first email address, last name, "
        "and the last 4 digits of the contact number. For example, 'surya1297pradeep3210'."
    ),
)
name_schema = ResponseSchema(
    name="Name",
    description="The full name of the candidate as a string.",
)

current_designation_schema = ResponseSchema(
    name="Current_Designation",
    description="The current job title or designation of the candidate as a string.",
)
email_schema = ResponseSchema(
    name="Email_Address",
    description="A list of email addresses of the candidate, with each email as a string.",
)

linkedin_schema = ResponseSchema(
    name="LinkedIn_URL",
    description="The LinkedIn profile URL of the candidate as a string.",
)
contact_number_schema = ResponseSchema(
    name="Contact_Number",
    description="A list of contact numbers with country codes prefixed, e.g., '+91-9876543210'.",
)
total_experience_schema = ResponseSchema(
    name="Total_Work_Experience_Days",
    description="The total number of days of work experience as an integer.",
)
skills_schema = ResponseSchema(
    name="Skills",
    description="A list of up to 15 skills of the candidate.",
)
score_schema = ResponseSchema(
    name="Score",
    description="The similarity score between the resume and job description.",
)
file_path_Schema=ResponseSchema(
    name="file_path",
    description="The path of the file in the folder",
)
# Compile all schemas into a list or dictionary for further use
schemas = [uid_schema, name_schema, current_designation_schema, email_schema,
          linkedin_schema, contact_number_schema,
          total_experience_schema, skills_schema,score_schema,file_path_Schema]



class Resume_details(BaseModel):
    UID: str
    Name: str
    Current_Designation: str
    Email_Address: List[str]
    LinkedIn_URL: str
    Contact_Number: int
    Total_Work_Experience_Days: int
    Skills: List[str]
    Score: int
    file_path: str

# Define a Pydantic model for MCQs with multiple questions
class mcqItem(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class mcqs(BaseModel):
    questions: List[mcqItem]  # List of multiple questions

output_parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = output_parser.get_format_instructions()
format_instructions

def process_pdf(pdf_path):
    all_info = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
            all_info.append(extracted_text)
        return all_info

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


# Define the first agent for resume data extraction
extractor = Agent(
    role="Resume Data Extractor",
    goal="Takes the {resume_data} and {job_description} and extracts the data in the form of {format_instructions}",
    backstory=(
        # "Take the input of file paths for resumes and job description and send to the tool {tools} to extract the resume_data and job_description and use them in the agent"
        "You're working on getting candidate resumes "
        "and extracting the information from {resume_data} in the format of {format_instructions}. "
        "Then score the {resume_data} based on the {job_description} between 0 to 10. "
        # "Add the score value obtained in the output of {resume_data} in the format of {format_instructions}."
        # "Even if the resume appears to be in the desired format, you must still extract the relevant information and format it accordingly." # Added instruction
    ),
    #   llm = huggingface_llm,
    # tools=[file_read_tool],
    allow_delegation=False,
    verbose=True
)
plan = Task(
    description=(
        "1. Parse the {resume_data} to identify and extract relevant sections "
        "like personal details, education, work experience, skills, certifications, etc.\n"
        "2. Match the extracted sections against the criteria specified in the {job_description}.\n"
        "3. Evaluate the relevance and completeness of the {resume_data} "
        "in the context of the {job_description} and assign a score between 0 to 10.\n"
        "4. Format the extracted information, along with the assigned score, "
        # "in the structure specified by the {format_instructions}."
        "**Important:** Your output MUST be a valid JSON string that adheres to the following structure: \n" # Emphasize JSON output

    ),
    expected_output=(
        "An extracted and formatted output of the {resume_data} "
        "with all relevant sections included and a score "
        "indicating the alignment with the {job_description}."
    ),
    output_json=Resume_details,
    output_file="Resume_details.json",
    agent=extractor,
)

# Define the second agent for MCQ generation
mcq_generator = Agent(
    role="MCQ Creator",
    goal="Generate conceptual MCQs based on the skills extracted from the resume and the job description.",
    backstory=(
        "You're an expert interviewer. Based on the candidate's skills mentioned in the resume and the requirements "
        "in the job description, generate conceptual MCQs to assess their understanding of relevant technologies."
    ),
    #    llm = huggingface_llm,

    allow_delegation=False,
    verbose=True
)

# Define the plan for the MCQ generation agent
plan_mcq_generator = Task(
    description=(
        "1. Take the extracted skills from the resume and the job description.\n"
        "2. Generate 5 conceptual MCQs that assess the candidate's understanding of these skills and technologies.\n"
        "3. Each question should have 4 options with one correct answer. "
        "Avoid using specific details from the resume in the questions or options.\n"
        "4. Provide the output in the specified format."
    ),
    expected_output=(
        "A list of 5 multiple-choice questions (MCQs) with 4 options each, "
        "indicating the correct answer explicitly."
    ),
    output_json=mcqs,
    output_file="mcqs.json",
    agent=mcq_generator,
)

def load_resume_details(json_path):
    """Loads resume details from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            resume_details = json.load(f)
        return resume_details
    else:
        print("Resume_details.json not found!")
        return None

FOLDER_PATH = input("Enter the resume folder path")
JOB_DESCRIPTION_PATH = input("Enter the job description file path")
CREDENTIALS_FILE = "/workspace/HR_Agent/cred2.json"
SHEET_ID = "1zcty8XBFed9IRy-mfpliR4fJBr3812v1RIvKppl60XA"
SHEET_RANGE = "A1"
SENDER_EMAIL = "kusumonika033@gmail.com"
SENDER_PASSWORD = "uroc bdxy vmxg ggbk"
FORM_TITLE = "MCQ Assessment Form"
# Initialize Google Agent
google_agent = GoogleAgent(CREDENTIALS_FILE)

# Process job description
job_description = process_pdf(JOB_DESCRIPTION_PATH)[0]


# llm = LLM(
#     model="openai/gpt-3.5-turbo", # call model by provider/model_name
#     temperature=0.0,
# )
llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.3
)
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(FOLDER_PATH, filename)
        resume_data = process_pdf(pdf_path)[0]  # Extract resume text

        # Prepare inputs for Crew
        inputs_extractor = {
            "resume_data": resume_data,
            "job_description": job_description,
            "format_instructions": "Extract key skills, match with job description, and compute a score."
        }

        crew = Crew(
            llm=llm,
            agents=[extractor, mcq_generator],
            tasks=[plan, plan_mcq_generator],
            verbose=True
        )

        # Process resume and generate results
        result_extractor = crew.kickoff(inputs=inputs_extractor)
        print(result_extractor)
        RESUME_JSON_PATH = "Resume_details.json"
        resume_details = load_resume_details(RESUME_JSON_PATH)

        # Save results to Google Sheets
        google_agent.save_to_google_sheets(SHEET_ID, SHEET_RANGE, resume_details)
        # If score > 7, send an email
        name = resume_details["Name"]
        score = resume_details["Score"]
        email = resume_details["Email_Address"][0]
        print(name,email,score)
        if score > 7:
            form_title = "MCQ Assessment Form"
            MCQS_JSON_PATH = "mcqs.json"
            mcqs_questions = load_resume_details(MCQS_JSON_PATH)
            form_url = google_agent.create_google_form(form_title, mcqs_questions["questions"])

            email_body1 = f"Candidate {name} has a score of {score}. Please review their resume: {pdf_path}"

            google_agent.send_email(
                to_email=SENDER_EMAIL,
                subject="High Matching Resume Alert",
                body=email_body1,
                sender_email=SENDER_EMAIL,
                sender_password=SENDER_PASSWORD
            )
            email_body = (f"Dear {name},\n\n"
                  f"Congratulations! You have been shortlisted for the first round of the interview.\n"
                  f"Your score: {score}.\n\n"
                  f"Please complete the assessment using the following form link: {form_url}\n\n"
                  f"Best regards,\n"
                  f"The Recruitment Team")
            email_subject = "You are shortlisted for the first round of the interview!"
            # Send email to the candidate
            print(email)
            google_agent.send_email(email, email_subject, email_body, SENDER_EMAIL, SENDER_PASSWORD)


print("Processing completed.")