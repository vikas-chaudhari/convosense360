import google.generativeai as genai
import os
import streamlit as st
import cohere  # Fixed import
from dotenv import load_dotenv
load_dotenv()
import json


def summarize(data):
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    print(gemini_api_key)
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    content = "Summarize the given chat file data below in detailed format without any loss of event : {data}".format(data=data)
    response = model.generate_content(content)
    print(response.text)
    return response.text

######====================================================

def textual_analysis(stats):
    # **1. Don't hard‑code your API key!** Instead, use environment variable CO_API_KEY.
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Please set COHERE_API_KEY environment variable.")

    # **2. Use the standard Cohere client:**
    co = cohere.Client(api_key=api_key)  # Standard client initialization

    prompt = f"""
    You are a chat analysis assistant. Based on the following WhatsApp chat insights, generate:
  1. A detailed 4–6 sentence summary, focusing on both positive and negative communication trends.
  2. Identify areas for improvement, and strategies to enhance communication.
  3. Applicable use‑cases in business, healthcare, personal development.
  4. Actionable next steps to improve group dynamics.
  5. A section "User Takeaways" with 8–10 practical steps for better communication and engagement.

    Insights:
    {json.dumps(stats, indent=2)}
    """

    try:
        response = co.chat(
            message=prompt,  # Using 'message' instead of 'messages' for v1 API
            model="command-r-plus",
            temperature=0.3,
            max_tokens=1024
        )
        # Response parsing for v1 API
        return response.text.strip()
    except Exception as e:
        return f"Error from Cohere: {e}"