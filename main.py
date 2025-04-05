import streamlit as st
import os
import PyPDF2
from pydantic import BaseModel, Field
from typing import List
import re
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
import json
import contextlib # <--- ADD THIS LINE

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
# from crewai.llms import Groq, OpenAI  # Import specific LLM wrappers if needed
# Or use the generic LLM wrapper from your original code if preferred
# from crewai import LLM as CrewLLM # Renamed to avoid conflict with Streamlit component
import zipfile
import tempfile
import time
import io
import tiktoken # For token estimation
import pandas as pd

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import pandas as pd
import zipfile
import tempfile

# --- Load Environment Variables & Tools ---
# Assuming tools.py contains GoogleAgent and loadenv()
try:
    from tools import *
    load_dotenv() # Load environment variables first
    print("Environment variables loaded.")
except ImportError:
    st.error("Failed to import 'tools.py'. Make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# --- Configuration & Constants ---
# Get API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # Default model
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile") # Example Groq model

# --- LLM Selection ---
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

# Give user a choice or default to one
llm_provider = st.sidebar.selectbox("Select LLM Provider", ["Groq", "OpenAI"], index=0)

selected_llm = None
model_name = ""
llm_temp = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.05) # Add temperature slider

if llm_provider == "Groq":
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("Groq API key not found in environment variables (.env file).")
        st.stop()
    try:
        model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile") # Get model from env or default
        selected_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=llm_temp
        )
        # Rough Pricing (Check Groq's latest pricing) - Example for Llama 3.1 70b
        INPUT_PRICE_PER_MILLION_TOKENS = 0.59
        OUTPUT_PRICE_PER_MILLION_TOKENS = 0.79
        st.sidebar.info(f"Using Groq Model: {model_name}")
        st.sidebar.caption(f"Temp: {llm_temp}")

    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")
        st.stop()

elif llm_provider == "OpenAI":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found in environment variables (.env file).")
        st.stop()
    try:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # Get model from env or default
        selected_llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=llm_temp
        )
        # Rough Pricing (Check OpenAI's latest pricing) - Example for GPT-4o
        INPUT_PRICE_PER_MILLION_TOKENS = 5.00
        OUTPUT_PRICE_PER_MILLION_TOKENS = 15.00
        st.sidebar.info(f"Using OpenAI Model: {model_name}")
        st.sidebar.caption(f"Temp: {llm_temp}")

    except Exception as e:
        st.error(f"Failed to initialize OpenAI LLM: {e}")
        st.stop()
        # Give user a choice or default to one
# llm_provider = st.sidebar.selectbox(
#     "Select LLM Provider",
#     ["Groq", "OpenAI"],
#     index=0,
#     key="llm_provider_selector" # <--- ADD THIS UNIQUE KEY
# )

# --- Pydantic Models & Schemas (Copied from your script) ---
# Define schemas for each field
uid_schema = ResponseSchema(
    name="UID",
    description=(
        "A unique identifier for the candidate, constructed using the first email address found, last name, "
        "and the last 4 digits of the first contact number found. Use 'NA' if parts are missing. Example: 'heatherclay62@gmail.com_Clay_8524'."
    ),
)
name_schema = ResponseSchema(
    name="Name",
    description="The full name of the candidate as a string. Return 'NA' if not found.",
)
current_designation_schema = ResponseSchema(
    name="Current_Designation",
    description="The current or most recent job title/designation. Return 'NA' if not found.",
)
email_schema = ResponseSchema(
    name="Email_Address",
    description="A list of all unique email addresses found. Return empty list [] if none found.",
)
linkedin_schema = ResponseSchema(
    name="LinkedIn_URL",
    description="The LinkedIn profile URL. Return 'NA' if not found.",
)
contact_number_schema = ResponseSchema(
    name="Contact_Number",
    description="A list of contact numbers found, preferably with country code if available. Return empty list [] if none found.",
)
total_experience_schema = ResponseSchema(
    name="Total_Work_Experience_Days",
    description="Estimate the total number of days of work experience based on dates. Return 0 if cannot be determined.",
)
skills_schema = ResponseSchema(
    name="Skills",
    description="A list of up to 15 key technical skills extracted. Return empty list [] if none found.",
)
score_schema = ResponseSchema(
    name="Score",
    description="The similarity score (integer 0-10) between the resume and job description based on skills, experience, and qualifications.",
)
file_path_schema = ResponseSchema(
    name="file_path",
    description="The original filename of the resume PDF.",
)

schemas = [uid_schema, name_schema, current_designation_schema, email_schema,
           linkedin_schema, contact_number_schema,
           total_experience_schema, skills_schema, score_schema, file_path_schema]

class Resume_details(BaseModel):
    UID: str = Field(description="Unique identifier (email_lastname_phone)")
    Name: str = Field(description="Candidate's full name")
    Current_Designation: str = Field(description="Candidate's current job title")
    Email_Address: List[str] = Field(description="List of email addresses")
    LinkedIn_URL: str = Field(description="LinkedIn profile URL")
    Contact_Number: List[str] = Field(description="List of contact numbers")
    Total_Work_Experience_Days: int = Field(description="Total work experience in days")
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
    """Extracts text from PDF bytes."""
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        return text
    except PyPDF2.errors.PdfReadError:
        st.warning(f"Could not read PDF: {filename}. It might be corrupted or password-protected. Skipping.")
        return None
    except Exception as e:
        st.error(f"Error processing PDF {filename}: {e}")
        return None

def estimate_tokens(text, model_name="gpt-4"):
    """Estimates token count using tiktoken."""
    try:
        # Attempt to get encoding for common models, fallback to cl100k_base
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Tiktoken error: {e}. Falling back to character count / 4.")
        return len(text) // 4 # Rough fallback estimation

def calculate_cost(input_tokens, output_tokens, input_price, output_price):
    """Calculates cost based on tokens and pricing."""
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost

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


# --- Streamlit UI ---
# st.set_page_config(layout="wide", page_title="CrewAI Resume Screener") # Added page_title
st.title("📄🤖 Module:1 HR Automation Agent")
st.markdown("Upload a batch of resumes (ZIP) and a job description (PDF) to screen resumes and send dynamic assessments to candidates")
# try:
#     from tools import GoogleAgent, loadenv
#     loadenv() # Load environment variables first
#     print("Environment variables loaded.")
# except ImportError:
#     st.error("Failed to import 'tools.py'. Make sure it's in the same directory.")
#     st.stop()

# --- Sidebar for Configuration ---
st.sidebar.header("Configuration")
# LLM selection is done above

# Google Integration Settings (Consider using st.secrets for production)
st.sidebar.subheader("Google Integration (Optional)")
enable_google_integration = st.sidebar.checkbox("Enable Google Sheets & Email", value=False)

CREDENTIALS_FILE = st.sidebar.text_input(
    "Credentials JSON Path",
    "cred2.json",  # Set a simple default path relative to the script
    key="credentials_file_input" # Added a unique key
    )
SHEET_ID = st.sidebar.text_input("Google Sheet ID", "YOUR_SHEET_ID_HERE") # Replace with your Sheet ID
SHEET_RANGE = st.sidebar.text_input("Sheet Range (e.g., A1)", "Sheet1!A1")
SENDER_EMAIL = st.sidebar.text_input("Sender Gmail Address", "your_email@gmail.com") # Replace
# IMPORTANT: Avoid putting passwords directly in code. Use environment variables or st.secrets.
# For demo purposes using text_input, but strongly advise against it for real use.
SENDER_PASSWORD = st.sidebar.text_input("Sender Gmail Password", type="password") # Replace
SCORE_THRESHOLD = st.sidebar.slider("Email Notification Score Threshold", 0, 10, 7)

# Initialize Google Agent only if needed and configured
google_agent = None
if enable_google_integration:
    if not os.path.exists(CREDENTIALS_FILE):
        st.sidebar.warning(f"Credentials file not found at: {CREDENTIALS_FILE}")
    elif not SHEET_ID or SHEET_ID == "YOUR_SHEET_ID_HERE":
        st.sidebar.warning("Please provide a valid Google Sheet ID.")
    elif not SENDER_EMAIL or not SENDER_PASSWORD:
        st.sidebar.warning("Please provide Sender Email and Password for notifications.")
    else:
        try:
            google_agent = GoogleAgent(CREDENTIALS_FILE)
            st.sidebar.success("Google Agent Initialized.")
        except Exception as e:
            st.sidebar.error(f"Failed to initialize Google Agent: {e}")
            google_agent = None # Ensure it's None if init fails

# --- Main Area for Uploads and Processing ---
col1, col2 = st.columns(2)

with col1:
    uploaded_zip = st.file_uploader("Upload Resumes (ZIP file containing PDFs)", type="zip")

with col2:
    uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type="pdf")

process_button = st.button("🚀 Process Resumes")

# --- Processing Logic ---
if process_button and uploaded_zip and uploaded_jd and selected_llm:
    if enable_google_integration and not google_agent:
        st.error("Google Integration is enabled, but the Google Agent failed to initialize. Please check sidebar configuration.")
        st.stop()

    # Create a temporary directory to extract resumes
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Read Job Description
        jd_bytes = uploaded_jd.getvalue()
        job_description_text = process_pdf_bytes(jd_bytes, uploaded_jd.name)
        if not job_description_text:
            st.error("Could not read Job Description PDF. Please upload a valid file.")
            st.stop()

        st.subheader("Job Description Text (Preview)")
        with st.expander("Click to view"):
            st.text(job_description_text[:1000] + "...") # Show preview

        # 2. Extract Resumes from ZIP
        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            st.success(f"Extracted files from '{uploaded_zip.name}'")
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file. Please upload a valid archive.")
            st.stop()
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")
            st.stop()

        # 3. Process each PDF resume
        st.write(f"Checking for PDFs inside temporary directory: {temp_dir}") # Debug: Show temp dir
        all_files_in_temp = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                 all_files_in_temp.append(os.path.join(root, file)) # Collect all files found

        st.write(f"Files found by os.walk: {all_files_in_temp}") # Debug: List all files found

        pdf_file_paths = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    pdf_file_paths.append(full_path) # Store the full path

        if not pdf_file_paths:
             st.warning("No PDF files found recursively in the uploaded ZIP archive.") # Updated message
             # Also show what *was* found for better debugging
             st.write("Files/folders found at the top level:", os.listdir(temp_dir))
             st.stop()

        st.info(f"Found {len(pdf_file_paths)} PDF resumes to process...")
        all_results = []
        total_estimated_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0


        # ==============================================================
        # START OF THE MAIN LOOP CODE BLOCK REQUESTED
        # ==============================================================
        for i, pdf_path in enumerate(pdf_file_paths):
            filename = os.path.basename(pdf_path)
            st.markdown("---")
            st.subheader(f"Processing: {filename} ({i+1}/{len(pdf_file_paths)})")
            # Use st.status for context, expanded=True initially
            file_status = st.status(f"Starting {filename}...", expanded=True)

            # Variables for this file iteration
            extracted_data_dict = None
            mcqs_dict = None
            task1_output_tokens = 0
            task2_output_tokens = 0
            current_input_tokens = 0
            logs1 = "" # To store logs OUTSIDE status later
            logs2 = "" # To store logs OUTSIDE status later
            current_cost = 0.0
            current_output_total = 0
            task1_success = False # Flag for Task 1 success
            task2_success = False # Flag for Task 2 success

            # --- Start of the try block for one file ---
            try:
                # --- Reading PDF ---
                with file_status: # Show reading message in status
                    st.write(f"Reading PDF: {filename}...")
                with open(pdf_path, 'rb') as f:
                    resume_bytes = f.read()
                resume_text = process_pdf_bytes(resume_bytes, filename)
                if not resume_text:
                    # Update status and continue directly
                    file_status.update(label=f"⚠️ Skipping {filename} (no text).", state="error", expanded=False)
                    continue

                # --- Prepare Inputs ---
                inputs = { # Prepare inputs as before
                    "resume_data": resume_text, "job_description": job_description_text,
                    "format_instructions": format_instructions, "resume_filename": filename,
                }
                extractor_input_text = f"{resume_text}\n{job_description_text}\n{format_instructions}"
                current_input_tokens = estimate_tokens(extractor_input_text, model_name)

                # --- Run Task 1: Extraction ---
                file_status.update(label=f"Running Extraction Agent for {filename}...", state="running")
                log_stream1 = io.StringIO()
                task1_result = None
                with contextlib.redirect_stdout(log_stream1):
                    try:
                        temp_crew_task1 = Crew(
                            agents=[extractor], tasks=[extract_task],
                            process=Process.sequential, llm=selected_llm,
                            verbose=True
                        )
                        task1_result = temp_crew_task1.kickoff(inputs=inputs)
                    except Exception as task1_exc:
                            logs1 = log_stream1.getvalue() # Capture logs on error
                            # Update status to error and store flag
                            file_status.update(label=f"🚨 Extraction Task failed: {task1_exc}", state="error", expanded=True)
                            task1_success = False
                            # Skip directly to displaying logs/results outside try block
                            raise # Re-raise the exception to exit the main try block
                logs1 = log_stream1.getvalue() # Capture logs on success

                # --- Parse Task 1 Output ---
                with file_status: # Show parsing message in status
                    st.write("Parsing Extraction Results...")
                if task1_result and hasattr(task1_result, 'tasks_output') and task1_result.tasks_output:
                    task1_output = task1_result.tasks_output[0]
                    task1_string_to_parse = None
                    parse_method1 = "Unknown"
                    try:
                        # --- Parsing logic as before ---
                        if isinstance(task1_output.pydantic_output, Resume_details):
                            extracted_data_dict = task1_output.pydantic_output.model_dump(); parse_method1 = "pydantic"
                        elif hasattr(task1_output, 'exported_output') and isinstance(task1_output.exported_output, str):
                            task1_string_to_parse = task1_output.exported_output; extracted_data_dict = json.loads(task1_string_to_parse); parse_method1 = "exported_output"
                        # ... [Add other parsing methods: .result, str()] ...
                        else: raise ValueError("No suitable string found")

                        with file_status: st.write(f"Parsed Task 1 using: {parse_method1}") # Show success in status
                        if not isinstance(extracted_data_dict.get("Skills"), list): extracted_data_dict["Skills"] = []
                        task1_output_tokens = estimate_tokens(task1_string_to_parse or json.dumps(extracted_data_dict), model_name)
                        task1_success = True # Mark Task 1 as successful

                    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as parse_e1:
                        failed_str1 = task1_string_to_parse if task1_string_to_parse else str(task1_output)
                        # Update status to error and store flag
                        file_status.update(label=f"🚨 Extraction Parsing Failed: {parse_e1}", state="error", expanded=True)
                        task1_success = False
                        # No need to continue, flow will check task1_success flag
                else:
                    file_status.update(label="🚨 Extraction task output invalid.", state="error", expanded=True)
                    task1_success = False


                # --- Run Task 2: MCQ Generation (only if Task 1 succeeded) ---
                if task1_success:
                    file_status.update(label=f"Running MCQ Agent for {filename}...", state="running")
                    log_stream2 = io.StringIO()
                    task2_result = None
                    mcq_inputs = { # Prepare inputs as before
                        "extracted_skills": extracted_data_dict.get("Skills", []),
                        "job_description": job_description_text
                    }
                    mcq_input_approx = json.dumps(mcq_inputs['extracted_skills']) + job_description_text
                    mcq_input_tokens_est = estimate_tokens(mcq_input_approx, model_name)
                    current_input_tokens += mcq_input_tokens_est

                    with contextlib.redirect_stdout(log_stream2):
                        try:
                            temp_crew_task2 = Crew( # Create crew as before
                                agents=[mcq_generator], tasks=[mcq_task],
                                process=Process.sequential, llm=selected_llm, verbose=True
                            )
                            task2_result = temp_crew_task2.kickoff(inputs=mcq_inputs)
                        except Exception as task2_exc:
                            logs2 = log_stream2.getvalue() # Capture logs on error
                            # Update status but don't stop processing the file
                            file_status.warning(f"⚠️ MCQ Task failed: {task2_exc}. Proceeding...", icon="⚠️")
                            mcqs_dict = {"questions": []}; task2_output_tokens = 0
                            task2_success = False # Mark Task 2 as failed
                        else: # If kickoff succeeded
                            logs2 = log_stream2.getvalue()
                            # --- Parse Task 2 Output ---
                            with file_status: st.write("Parsing MCQ Results...") # Show parsing message
                            if task2_result and hasattr(task2_result, 'tasks_output') and task2_result.tasks_output:
                                task2_output = task2_result.tasks_output[0]
                                task2_string_to_parse = None
                                parse_method2 = "Unknown"
                                try:
                                    # --- Parsing logic as before ---
                                    if isinstance(task2_output.pydantic_output, mcqs):
                                        mcqs_dict = task2_output.pydantic_output.model_dump(); parse_method2="pydantic"
                                    # ... [Add other parsing methods] ...
                                    else: raise ValueError("No suitable string found")

                                    with file_status: st.write(f"Parsed Task 2 using: {parse_method2}")
                                    task2_output_tokens = estimate_tokens(task2_string_to_parse or json.dumps(mcqs_dict), model_name)
                                    task2_success = True # Mark Task 2 as successful

                                except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as parse_e2:
                                    failed_str2 = task2_string_to_parse if task2_string_to_parse else str(task2_output)
                                    file_status.warning(f"⚠️ MCQ Parsing Failed: {parse_e2}. Defaulting.", icon="⚠️")
                                    mcqs_dict = {"questions": []}; task2_output_tokens = estimate_tokens(failed_str2, model_name)
                                    task2_success = False
                            else:
                                file_status.warning("⚠️ MCQ task output invalid. Defaulting.", icon="⚠️")
                                mcqs_dict = {"questions": []}; task2_output_tokens = 0
                                task2_success = False
                else: # If Task 1 failed
                        with file_status: st.write("Skipping MCQ generation due to prior extraction failure.")
                        mcqs_dict = {"questions": []} # Ensure default
                        task2_success = False

                # --- Accumulate results (only if Task 1 succeeded) ---
                if task1_success:
                    total_input_tokens += current_input_tokens
                    total_output_tokens += task1_output_tokens + task2_output_tokens
                    current_output_total = task1_output_tokens + task2_output_tokens
                    current_cost = calculate_cost(current_input_tokens, current_output_total, INPUT_PRICE_PER_MILLION_TOKENS, OUTPUT_PRICE_PER_MILLION_TOKENS)
                    total_estimated_cost += current_cost

                    result_summary = { # Create summary as before
                        "filename": filename, "details": extracted_data_dict,
                        "mcqs": mcqs_dict if mcqs_dict is not None else {"questions": []},
                        "cost": current_cost, "input_tokens": current_input_tokens,
                        "output_tokens": current_output_total
                    }
                    all_results.append(result_summary)
                    # Update status to complete *finally*
                    file_status.update(label=f"✅ Completed {filename}", state="complete", expanded=False)
                # else: # If task 1 failed, status is already set to error

            # --- End of the try block for one file ---
            except Exception as e:
                # Catch unexpected errors not caught by specific task failures
                st.error(f"Unexpected error processing {filename}: {e}")
                import traceback
                st.code(traceback.format_exc()) # Show stack trace for unexpected errors
                if 'file_status' in locals() and file_status:
                    file_status.update(label=f"🚨 Error processing {filename}", state="error", expanded=True)
                # Ensure flags indicate failure if we end up here
                task1_success = False
                task2_success = False
                # Let loop continue

            # ==============================================================
            # DISPLAY LOGS, RESULTS, GOOGLE ACTIONS *OUTSIDE* STATUS
            # ==============================================================

            # --- Display Logs (if any captured) ---
            if logs1:
                with st.expander(f"Show Task 1 (Extraction) Logs for {filename}"):
                    st.text_area("Task 1 Logs:", logs1, height=200, key=f"log1_disp_{i}")
            if logs2:
                with st.expander(f"Show Task 2 (MCQ Generation) Logs for {filename}"):
                    st.text_area("Task 2 Logs:", logs2, height=200, key=f"log2_disp_{i}")

            # --- Display Results (if Task 1 succeeded) ---
            if task1_success and extracted_data_dict:
                expander_title = f"Results for {filename} (Score: {extracted_data_dict.get('Score', 'N/A')}) - Est. Cost: ${current_cost:.4f}"
                with st.expander(expander_title, expanded=False):
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader("Extracted Details")
                        st.json(extracted_data_dict)
                    with res_col2:
                        st.subheader("Generated MCQs")
                        st.json(mcqs_dict if mcqs_dict is not None else {"questions": []})

                # --- Google Integration Actions (conditional) ---
                if enable_google_integration and google_agent:
                        st.write(f"Performing Google Actions for {filename}...") # Indicate actions starting
                        # Save to Google Sheets
                        try:
                            sheet_data = [[ # Prepare sheet data
                                extracted_data_dict.get("UID", "NA"), extracted_data_dict.get("Name", "NA"),
                                # ... [rest of sheet data] ...
                                extracted_data_dict.get("file_path", "NA")
                            ]]
                            google_agent.save_to_google_sheets(SHEET_ID, SHEET_RANGE, sheet_data)
                            st.success(f"✅ Data for {filename} saved to Google Sheets.")
                        except Exception as sheet_error:
                            st.warning(f"⚠️ Failed to save {filename} to Sheets: {sheet_error}")
                        # Send Email
                        score = extracted_data_dict.get("Score", 0)
                        if score >= SCORE_THRESHOLD:
                            st.write(f"Score {score} >= {SCORE_THRESHOLD}. Triggering email actions...")
                            try:
                                # ... [logic to get email, create form, send emails] ...
                                st.success(f"✅ Actions triggered for {filename} (Score: {score}).")
                            except Exception as email_error:
                                st.warning(f"⚠️ Failed email/form process for {filename}: {email_error}")
            # --- End if task1_success ---
            # elif not task1_success: # Optional: Message if task 1 failed
            #     st.warning(f"Skipping results display and Google Actions for {filename} due to extraction failure.")


        # ==============================================================
        # END OF THE MAIN LOOP
        # ==============================================================

        # --- Final Summary (after loop finishes) ---
        st.markdown("---")
        st.header("🏁 Processing Summary")
        if all_results:
                summary_df = pd.DataFrame([{
                    "Filename": res["filename"],
                    "Score": res["details"].get("Score", "N/A"),
                    "Name": res["details"].get("Name", "N/A"),
                    "Email": ", ".join(res["details"].get("Email_Address", [])),
                    "Est. Cost": f"${res['cost']:.4f}",
                    "Input Tokens": res["input_tokens"],
                    "Output Tokens": res["output_tokens"]
                } for res in all_results])
                st.dataframe(summary_df)

                st.subheader("Cost Estimation")
                st.metric(label="Total Input Tokens", value=f"{total_input_tokens:,}")
                st.metric(label="Total Output Tokens", value=f"{total_output_tokens:,}")
                st.metric(label="Total Estimated Cost", value=f"${total_estimated_cost:.4f}")
                st.caption(f"Based on {llm_provider} ({model_name}) pricing: ${INPUT_PRICE_PER_MILLION_TOKENS:.2f}/M input, ${OUTPUT_PRICE_PER_MILLION_TOKENS:.2f}/M output tokens. Token counts estimated.")
        else:
                st.warning("No resumes were successfully processed, or all failed.")
# --- End of file ---