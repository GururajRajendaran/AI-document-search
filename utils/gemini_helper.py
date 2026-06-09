import os
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    api_key = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(
    api_key=api_key
)

def ask_gemini(question, context):

    prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say 'Information not found in document'.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text