from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
gemini = GoogleGenerativeAI(  model="gemini-3-flash-preview")
st.header('DEMO' )

user_input = st.text_input('Enter Your Prompt')


if st.button('Submit'):
         result = gemini.invoke(user_input)
         st.write(result)


