from langchain_google_genai  import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate #dynamic prompt

load_dotenv()

gemini = GoogleGenerativeAI(  model="gemini-3-flash-preview")

st.header(" Kids Story Generator")

# User Inputs
age = st.number_input("Enter Child Age", min_value=3, max_value=15, step=1)

gender = st.selectbox(
    "Select Gender",
    ["Boy", "Girl", "Other"]
)

genre = st.selectbox(
    "Select Story Genre",
    ["Fantasy", "Adventure", "Moral Story", "Fairy Tale", "Sci-Fi"]
)

length = st.selectbox(
    "Select Story Length",
    ["Short", "Medium", "Long"]
)

story_prompt = PromptTemplate(
    input_variables=["age", "gender", "genre", "length"],

    template="""
You are a creative children's story writer.

Write a {length} {genre} story for a {age}-year-old {gender}.
The language should be age-appropriate, engaging, fun, and imaginative.
Include positive values and a happy ending.
""",
 validate_template= True,
)

if st.button(" Generate Story"):
    #the output of the prompt becomes the input to the model.
    chain = story_prompt | gemini

    result = chain.invoke({
        "age": age,
        "gender": gender,
        "genre": genre,
        "length": length
    })

    st.write(result)