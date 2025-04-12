from dotenv import load_dotenv
import os
import google.generativeai as genai
import streamlit as st
load_dotenv()

gemini_api_key = os.environ.get('GEMINI_API_KEY')
print(gemini_api_key)

def summarize(data):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    content = "Summarize the given chat file data below in detailed format without any loss of event : {data}".format(data=data)
    response = model.generate_content(content)
    print(response.text)
    return response.text
     