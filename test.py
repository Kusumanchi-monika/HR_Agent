# --- Start of file ---
import streamlit as st
import os
import PyPDF2
from pydantic import BaseModel, Field
from typing import List
import re
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
import json
import contextlib # Keep for potential future debugging, but not used for UI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import zipfile
import tempfile
import time
import io
import tiktoken
import pandas as pd
import traceback # For logging unexpected errors

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tools import *

# --- Must be the first Streamlit command ---
st.set_page_config(layout="wide", page_title="CrewAI Resume Screener")

# --- Load Environment Variables & Tools ---
try:
    from tools import *
    load_dotenv()
    print("Environment variables loaded.")
except ImportError:
    st.warning("Could not import from 'tools.py'. Google Integration will be disabled.")
    class GoogleAgent: # Define dummy class if tools.py is missing
        def __init__(self, creds): pass
        def save_to_google_sheets(self, sheet_id, sheet_range, data): print("GoogleAgent disabled: save_to_google_sheets"); pass
        def create_google_form(self, title, questions): print("GoogleAgent disabled: create_google_form"); return None
        def send_email(self, to_email, subject, body, sender_email, sender_password): print(f"GoogleAgent disabled: send_email to {to_email}"); pass
    def loadenv(): pass
    load_dotenv()
except Exception as e:
    st.error(f"Error loading environment variables or tools.py: {e}")
    st.stop()

# --- Configuration & Constants ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLM Selection ---
st.sidebar.header("Configuration")
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider", ["Groq", "OpenAI"], index=0, key="llm_provider_selector"
)
selected_llm = None
model_name = ""
INPUT_PRICE_PER_MILLION_TOKENS = 0.0
OUTPUT_PRICE_PER_MILLION_TOKENS = 0.0
llm_temp = st.sidebar.slider(
    "LLM Temperature", 0.0, 1.0, 0.3, 0.05, key="llm_temp_slider"
)

# --- Initialize LLM ---
if llm_provider == "Groq":
    if not GROQ_API_KEY: st.sidebar.error("Groq API key missing.")
    else:
        try:
            model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")
            selected_llm = ChatGroq(api_key=GROQ_API_KEY, model_name=model_name, temperature=llm_temp)
            INPUT_PRICE_PER_MILLION_TOKENS, OUTPUT_PRICE_PER_MILLION_TOKENS = 0.59, 0.79
            st.sidebar.info(f"Using Groq: {model_name}")
        except Exception as e: st.sidebar.error(f"Groq Init Error: {e}"); selected_llm = None
elif llm_provider == "OpenAI":
    if not OPENAI_API_KEY: st.sidebar.error("OpenAI API key missing.")
    else:
        try:
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
            selected_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name, temperature=llm_temp)
            INPUT_PRICE_PER_MILLION_TOKENS, OUTPUT_PRICE_PER_MILLION_TOKENS = 5.00, 15.00
            st.sidebar.info(f"Using OpenAI: {model_name}")
        except Exception as e: st.sidebar.error(f"OpenAI Init Error: {e}"); selected_llm = None

# --- Pydantic Models & Schemas ---
# (Keep your existing Pydantic/Schema definitions - they are needed for CrewAI tasks)
uid_schema = ResponseSchema(name="UID",description="Unique identifier (email_lastname_phone)")
name_schema = ResponseSchema(name="Name", description="Candidate's full name")
current_designation_schema = ResponseSchema(name="Current_Designation", description="Candidate's current job title")
email_schema = ResponseSchema(name="Email_Address", description="List of email addresses")
linkedin_schema = ResponseSchema(name="LinkedIn_URL", description="LinkedIn profile URL")
contact_number_schema = ResponseSchema(name="Contact_Number", description="List of contact numbers")
total_experience_schema = ResponseSchema(name="Total_Work_Experience_Days", description="Total work experience in days")
skills_schema = ResponseSchema(name="Skills", description="List of key skills")
score_schema = ResponseSchema(name="Score", description="Resume-JD match score (0-10)")
file_path_schema = ResponseSchema(name="file_path", description="Original filename")

schemas = [uid_schema, name_schema, current_designation_schema, email_schema,
           linkedin_schema, contact_number_schema,
           total_experience_schema, skills_schema, score_schema, file_path_schema]

class Resume_details(BaseModel):
    UID: str = Field(description="Unique identifier (email_lastname_phone)")
    Name: str = Field(description="Candidate's full name")
    Current_Designation: str = Field(description="Candidate's current job title")
    Email_Address: List[str] = Field(description="List of email addresses")
    LinkedIn_URL: str = Field(description="LinkedIn profile URL", default="NA")
    Contact_Number: List[str] = Field(description="List of contact numbers")
    Total_Work_Experience_Days: int = Field(description="Total work experience in days", default=0)
    Skills: List[str] = Field(description="List of key skills")
    Score: int = Field(description="Resume-JD match score (0-10)")
    file_path: str = Field(description="Original filename")

class mcqItem(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class mcqs(BaseModel):
    questions: List[mcqItem]

# --- Output Parsers ---
try:
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = output_parser.get_format_instructions()
except Exception as e:
    st.error(f"Error creating StructuredOutputParser: {e}")
    st.stop()

# --- Helper Functions ---
def process_pdf_bytes(pdf_bytes, filename="uploaded_pdf"):
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        if not reader.pages: return None
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        return text.strip() if text and text.strip() else None
    except PyPDF2.errors.PdfReadError as pdf_err:
        print(f"PDF Read Error ({filename}): {pdf_err}") # Log to console
        return None
    except Exception as e:
        print(f"PDF Processing Error ({filename}): {e}") # Log to console
        return None

def estimate_tokens(text, model_name="gpt-4"):
    if not text: return 0
    try:
        try: encoding = tiktoken.encoding_for_model(model_name)
        except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Tiktoken error: {e}. Fallback."); return len(text)//4

def calculate_cost(inp, outp, iprice, oprice): return (inp/1e6)*iprice + (outp/1e6)*oprice

# --- CrewAI Agent and Task Definitions ---
# (Keep agent/task definitions as before, but verbose can be False)
# --- CrewAI Agent and Task Definitions ---
# Define agents (make sure LLM is passed during Crew execution or initialization)
extractor = Agent(
    role="Resume Data Extractor",
    goal="Extract structured information from the resume text ({resume_data}) according to the format instructions, and score its relevance against the job description ({job_description}).",
    backstory=(
        "You are an expert HR assistant specializing in parsing resumes. "
        "Your task is to meticulously extract key details from a candidate's resume text. "
        "Then, compare the extracted skills and experience against the provided job description text. "
        "Assign an integer score from 0 (no match) to 10 (perfect match) based on this comparison. "
        "Strictly follow the provided JSON format instructions for the output, including the filename and the calculated score. Ensure the output is a single, valid JSON object."
        "Mandatory fields are Name, Email_Address, Contact_Number, Skills, Score, file_path. Fill with 'NA' or [] or 0 if information is missing."
        "Generate the UID as specified: first_email_lastname_last4digits_of_first_phone. Use 'NA' for missing parts."
    ),
    # llm=selected_llm, # Pass LLM here or during kickoff
    allow_delegation=False,
    verbose=False # Keep internal verbose off for cleaner Streamlit output
)

mcq_generator = Agent(
    role="Technical Interview Question Creator",
    goal="Generate 5 conceptual Multiple-Choice Questions (MCQs) based on the key skills identified in the extracted resume data and the requirements in the job description.",
    backstory=(
        "You are an experienced technical interviewer. Based on a candidate's skills (provided as input) "
        "and the job requirements (also provided), create 5 insightful MCQs. "
        "These questions should test fundamental understanding, not specific project details from the resume. "
        "Each MCQ must have exactly 4 options, with one clearly correct answer. Format the output as a JSON list of questions."
    ),
    # llm=selected_llm, # Pass LLM here or during kickoff
    allow_delegation=False,
    verbose=False # Keep internal verbose off
)

# Define tasks (inputs will be dynamically added)
extract_task = Task(
    description=(
        "1. Parse the provided {resume_data}.\n"
        "2. Identify and extract Personal Details (Name, Email, Phone, LinkedIn), Work Experience (Current Designation, Total Duration), and Skills.\n"
        "3. Compare the extracted information, especially skills and experience duration, against the {job_description}.\n"
        "4. Calculate a relevance Score (0-10).\n"
        "5. Construct the UID from the first email, last name, and last 4 digits of the first phone number found.\n"
        "6. Format the extracted data, UID, Score, and the original {resume_filename} strictly according to the following JSON structure:\n{format_instructions}"
    ),
    expected_output=(
        "A single, valid JSON object conforming to the Resume_details model, containing the extracted information, UID, calculated score, and original filename."
    ),
    output_json=Resume_details,
    # output_file="Resume_details_temp.json", # Optional: writes to file if needed
    agent=extractor,
    human_input=False
)

mcq_task = Task(
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
    # output_file="mcqs_temp.json", # Optional: writes to file if needed
    agent=mcq_generator,
    context=[extract_task], # Use context from the previous task
    human_input=False
)
# --- Sidebar Google Integration ---
st.sidebar.subheader("Google Integration (Optional)")
enable_google_integration = st.sidebar.checkbox("Enable Google Sheets & Email", value=False, key="google_integration_checkbox")
CREDENTIALS_FILE = st.sidebar.text_input("Credentials JSON Path", "cred2.json", key="credentials_file_input", disabled=not enable_google_integration)
SHEET_ID = st.sidebar.text_input("Google Sheet ID", "1zcty8XBFed9IRy-mfpliR4fJBr3812v1RIvKppl60XA", key="sheet_id_input", disabled=not enable_google_integration)
SHEET_RANGE = st.sidebar.text_input("Sheet Range (e.g., Sheet1!A1)", "Sheet1!A1", key="sheet_range_input", disabled=not enable_google_integration)
SENDER_EMAIL = st.sidebar.text_input("Sender Gmail Address", "kusumonika033@gmail.com", key="sender_email_input", disabled=not enable_google_integration)
SENDER_PASSWORD = st.sidebar.text_input("Sender Gmail App Password", type="password", key="sender_password_input", disabled=not enable_google_integration)
SCORE_THRESHOLD = st.sidebar.slider("Email Threshold", 0, 10, 7, key="score_threshold_slider", disabled=not enable_google_integration)

google_agent = None
if enable_google_integration:
    if not SHEET_ID or SHEET_ID == "YOUR_SHEET_ID_HERE": st.sidebar.warning("Provide Sheet ID.")
    elif not os.path.exists(CREDENTIALS_FILE): st.sidebar.warning(f"Creds file missing: {CREDENTIALS_FILE}")
    elif not SENDER_EMAIL or not SENDER_PASSWORD: st.sidebar.warning("Provide Sender Email/Password.")
    else:
        try:
            google_agent = GoogleAgent(CREDENTIALS_FILE)
            st.sidebar.success("Google Agent Initialized.")
        except Exception as e: st.sidebar.error(f"Google Agent Error: {e}"); google_agent = None

# --- Main Area ---
st.title("📄🤖 CrewAI Resume Screening Pipeline")
st.markdown("Upload resumes (ZIP) and a job description (PDF). Results appear below after processing.")

col1, col2 = st.columns(2)
with col1: uploaded_zip = st.file_uploader("1. Upload Resumes ZIP", type="zip")
with col2: uploaded_jd = st.file_uploader("2. Upload Job Description PDF", type="pdf")

process_button = st.button("🚀 Process Resumes")

# --- Processing Logic ---
if process_button:
    # --- Initial Checks ---
    if not uploaded_zip or not uploaded_jd: st.warning("Please upload both files.")
    elif not selected_llm: st.error("LLM not initialized. Check API key/sidebar.")
    elif enable_google_integration and not google_agent: st.error("Google Integration enabled but failed.")
    else:
        # --- Start Processing ---
        st.info("Processing started... Please wait.") # Initial message
        progress_bar = st.progress(0.0, text="Initializing...") # Progress bar

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Read Job Description
            progress_bar.progress(0.01, text="Reading Job Description...")
            jd_bytes = uploaded_jd.getvalue()
            job_description_text = process_pdf_bytes(jd_bytes, uploaded_jd.name)
            if not job_description_text:
                st.error("Could not read Job Description PDF."); st.stop()

            # 2. Extract Resumes from ZIP
            progress_bar.progress(0.05, text="Extracting Resumes from ZIP...")
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref: zip_ref.extractall(temp_dir)
                print(f"Extracted files to {temp_dir}") # Console log
            except Exception as e: st.error(f"Error extracting ZIP: {e}"); st.stop()

            # 3. Find PDF files
            progress_bar.progress(0.1, text="Finding PDF files...")
            pdf_file_paths = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(".pdf") and not file.startswith('.'):
                        pdf_file_paths.append(os.path.join(root, file))

            if not pdf_file_paths: st.warning("No PDF files found in ZIP."); st.stop()
            num_files = len(pdf_file_paths)
            st.info(f"Found {num_files} PDF(s) to process.")

            all_results = []
            total_estimated_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            # ... (previous code remains the same up to the loop start) ...

            processed_files_count = 0
            failed_files = []

            # Loop through files in the specified folder
            if not os.path.isdir(FOLDER_PATH):
                print(f"ERROR: Folder not found: {FOLDER_PATH}. Exiting.")
                exit()

            print(f"Starting loop through files in {FOLDER_PATH}...")
            for i, filename in enumerate(os.listdir(FOLDER_PATH)):
                # Check if it's a PDF file first
                if not filename.lower().endswith(".pdf"):
                    continue # Skip non-PDF files

                pdf_path = os.path.join(FOLDER_PATH, filename)
                print(f"\n--- Processing file {i+1}: {filename} ---") # Use index for clarity

                # Reset variables for this file
                parsed_resume_details = None
                parsed_mcqs = None
                resume_details_from_file = None
                mcqs_from_file = None
                final_resume_details = None # Initialize final dicts to None
                final_mcqs = None
                task1_success = False
                task2_success = False

                # --- Start of the try block for this file ---
                try:
                    # 1. Read Resume PDF
                    print("Reading resume PDF...")
                    resume_data = process_pdf(pdf_path)
                    if not resume_data:
                        print(f"Skipping {filename} due to PDF read error or empty content.")
                        failed_files.append(filename + " (PDF Read Error)")
                        continue # Correctly placed: skip rest of try block for this file

                    # 2. Prepare Inputs for Crew
                    inputs = {
                        "resume_data": resume_data,
                        "job_description": job_description,
                        "format_instructions": format_instructions,
                        "resume_filename": filename,
                    }

                    # 3. Create and Run Crew
                    print("Initializing and running Crew...")
                    crew = Crew(
                        agents=[extractor, mcq_generator], tasks=[extract_task, mcq_task],
                        process=Process.sequential, llm=selected_llm, verbose=True
                    )
                    crew_result = None
                    try:
                        crew_result = crew.kickoff(inputs=inputs)
                        print("Crew execution finished.")
                    except Exception as crew_exc:
                        print(f"ERROR during Crew execution for {filename}: {crew_exc}")
                        print(traceback.format_exc())
                        failed_files.append(filename + " (Crew Execution Error)")
                        continue # Correctly placed: skip rest of try block for this file

                    # 4. Attempt to Parse Direct Crew Output
                    print("Attempting to parse direct Crew output...")
                    # ... (Keep the robust parsing logic for task1_output and task2_output here) ...
                    # ... (Set parsed_resume_details and parsed_mcqs) ...
                    if crew_result and hasattr(crew_result, 'tasks_output') and len(crew_result.tasks_output) >= 2:
                        task1_output = crew_result.tasks_output[0]
                        task2_output = crew_result.tasks_output[1]
                        # Parse Task 1 Direct Output
                        try:
                            if hasattr(task1_output, 'pydantic_output'): # ... rest of parsing logic ...
                                parsed_resume_details = task1_output.pydantic_output.model_dump()
                                print("Parsed Task 1 direct output")
                            # ... other parsing methods ...
                        except Exception as parse_e1: print(f"WARNING: Failed Task 1 direct parse: {parse_e1}")
                        # Parse Task 2 Direct Output
                        try:
                            if hasattr(task2_output, 'pydantic_output'): # ... rest of parsing logic ...
                                parsed_mcqs = # ...
                                print("Parsed Task 2 direct output")
                            # ... other parsing methods ...
                        except Exception as parse_e2: print(f"WARNING: Failed Task 2 direct parse: {parse_e2}")
                    else: print("WARNING: Crew result structure unexpected.")


                    # 5. Load Results from JSON Files
                    print("Loading results from output JSON files...")
                    RESUME_JSON_PATH = "Resume_details.json"
                    MCQS_JSON_PATH = "mcqs.json"
                    resume_details_from_file = load_json_file(RESUME_JSON_PATH)
                    mcqs_from_file = load_json_file(MCQS_JSON_PATH)

                    # --- Decide which data to use ---
                    final_resume_details = parsed_resume_details if parsed_resume_details else resume_details_from_file
                    final_mcqs = parsed_mcqs if parsed_mcqs else mcqs_from_file

                    # 6. Check if essential data was obtained
                    if not final_resume_details:
                        print(f"ERROR: Could not obtain resume details for {filename}.")
                        failed_files.append(filename + " (Data Extraction Failed)")
                        continue # Correctly placed: skip rest of try block for this file
                    else:
                        print("Resume Details Obtained.") # Confirmation
                        task1_success = True # Mark as success if we got the details

                    if not final_mcqs:
                        print(f"WARNING: Could not obtain MCQs for {filename}. Proceeding without.")
                        final_mcqs = {"questions": []} # Default
                        task2_success = False
                    else:
                        print("MCQs Obtained.")
                        task2_success = True


                    # 7. Perform Google Actions (if data obtained and agent available)
                    if task1_success and google_agent: # Check task1_success here
                        print("Performing Google Actions...")
                        # ... (Keep Google Sheets and Email logic here) ...
                        # ... (Make sure this logic uses final_resume_details and final_mcqs) ...
                        try:
                            # ... Save to sheets ...
                            print(f"Saved {filename} data to Google Sheets.")
                        except Exception as sheet_error: print(f"ERROR saving {filename} to Sheets: {sheet_error}")

                        score = final_resume_details.get("Score", 0)
                        if score >= SCORE_THRESHOLD # ... rest of email logic ...
                            print(f"Email actions for {filename} completed/attempted.")
                        else: print(f"Skipping email for {filename} (score {score} < {SCORE_THRESHOLD}).")
                    elif not google_agent:
                        print("Skipping Google Actions (Agent not initialized).")
                    elif not task1_success:
                        print("Skipping Google Actions (Data Extraction Failed).")


                    # If we reach here, the file was processed (maybe with warnings, but no critical failure)
                    processed_files_count += 1

                # --- Error Handling for the entire file's try block ---
                except Exception as file_proc_error:
                    print(f"!! UNEXPECTED ERROR processing file {filename}: {file_proc_error} !!")
                    print(traceback.format_exc())
                    failed_files.append(filename + " (Unexpected Error in Loop)")
                    # The loop will naturally continue to the next file

            # --- End of the for loop ---

            # --- Final Summary ---
            print("\n--- Processing Completed ---")
            # ... (rest of the summary printing) ...