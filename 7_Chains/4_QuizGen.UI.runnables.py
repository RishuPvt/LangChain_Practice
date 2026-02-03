import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#lets you run multiple chains / functions at the same time on the same input, and then collect all their outputs together.
from langchain_core.runnables import RunnableParallel


load_dotenv()

st.set_page_config(
    page_title="Notes & Quiz Generator",
)

st.title(" Notes & Quiz Generator")
st.write("Generate **short notes** and **quiz questions** from any text.")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

final_chain = parallel_chain | merge_chain


input_text = st.text_area(
    " Enter your text here",
    height=250,
    placeholder="Paste study material"
)

button = st.button("Generate Notes & Quiz")

if button:
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating content..."):
            result = final_chain.invoke({"text": input_text})

        st.markdown("#Generated Content")
        st.markdown(result)
