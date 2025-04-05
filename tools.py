from __future__ import annotations
import json
import gspread
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from typing import List

# Google Sheets and Forms Integration
class GoogleAgent:
    def __init__(self, credentials_file: str):
        self.credentials = Credentials.from_service_account_file(credentials_file, scopes=[
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/forms'
        ])

    def save_to_google_sheets(self, sheet_id: str, sheet_range: str, data: dict):
        """
        Saves dictionary data to Google Sheets as a single row.

        Args:
            sheet_id: The ID of the Google Sheet.
            sheet_range: The range where data should be appended (e.g., 'A1').
            data: The dictionary containing the data to be saved.
        """
        client = gspread.authorize(self.credentials)
        sheet = client.open_by_key(sheet_id).sheet1

        # Get the headers (keys) from the dictionary
        headers = list(data.keys())

        # Get the values corresponding to the headers
        values = [str(val) if isinstance(val, list) else val for val in data.values()]  # Updated

        # Check if the sheet is empty and add headers if necessary
        if sheet.row_count == 1:
            sheet.append_row(headers)  # Add header row

        # Append the values as a single row
        sheet.append_row(values)

    def create_google_form(self, form_title: str, questions: List[dict]):
        service = build('forms', 'v1', credentials=self.credentials)

        # 1. Create the form with only the title
        form_body = {
            "info": {
                "title": form_title,
            }
        }
        form = service.forms().create(body=form_body).execute()
        form_id = form["formId"]

        # 2. Add questions using batchUpdate
        update_body = {
            "requests": [
                {
                    "createItem": {
                        "item": {
                            "title": question["question"],
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": [
                                            {"value": option} for option in question["options"]
                                        ],
                                        "shuffle": False
                                    }
                                }
                            }
                        },
                        "location": {
                            "index": i  # Add questions sequentially
                        }
                    }
                } for i, question in enumerate(questions)
            ]
        }
        service.forms().batchUpdate(formId=form_id, body=update_body).execute()

        return form["responderUri"]

    def get_data_from_sheet(self, sheet_id: str):
        """
        Retrieves all data from the Google Sheet.

        Args:
            sheet_id: The ID of the Google Sheet.

        Returns:
            A list of dictionaries where each dictionary represents a row.
        """
        client = gspread.authorize(self.credentials)
        sheet = client.open_by_key(sheet_id).sheet1

        # Get all data from the sheet
        rows = sheet.get_all_records()
        return rows

    def send_email(self, to_email: str, subject: str, body: str, sender_email: str, sender_password: str):
        """
        Sends an email using an SMTP server.

        Args:
            to_email: Recipient's email address.
            subject: Subject of the email.
            body: Body of the email.
            sender_email: Sender's email address.
            sender_password: Sender's email password.
        """
        # Set up the MIME
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = to_email
        message['Subject'] = subject

        # Attach the body with the msg instance
        message.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server and send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)

# credentials_file = "/content/oval-machine-442411-p0-4209411bbcc9.json"
# sheet_id = "1zcty8XBFed9IRy-mfpliR4fJBr3812v1RIvKppl60XA"
# sheet_range = "A1"
# google_agent = GoogleAgent(credentials_file)
# form_title = "MCQ Assessment Form"
# sender_email = "kusumonika033@gmail.com"
# sender_password = "uroc bdxy vmxg ggbk"

# google_agent.save_to_google_sheets(sheet_id, sheet_range, resume_details)

# form_url = google_agent.create_google_form(form_title, mcqs_questions["questions"])

# print(f"Google Form URL: {form_url}")