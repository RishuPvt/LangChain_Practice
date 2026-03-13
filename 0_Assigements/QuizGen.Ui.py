import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(page_title="Notes & Quiz Generator")
st.title("Notes & Quiz Generator (PDF Upload)")
st.write("Upload a PDF and generate **short notes** and **quiz questions**")

# -------- Model --------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -------- Prompts --------
notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

easy_quiz_prompt = PromptTemplate(
    template="Generate 5 very simple beginner-level question answers from the following text:\n{text}",
    input_variables=["text"]
)

medium_quiz_prompt = PromptTemplate(
    template="Generate 5 conceptual medium-level question answers from the following text:\n{text}",
    input_variables=["text"]
)

hard_quiz_prompt = PromptTemplate(
    template="Generate 5 advanced analytical question answers from the following text:\n{text}",
    input_variables=["text"]
)

merge_prompt = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# -------- Chains --------
notes_chain = notes_prompt | model | parser
easy_quiz_chain = easy_quiz_prompt | model | parser
medium_quiz_chain = medium_quiz_prompt | model | parser
hard_quiz_chain = hard_quiz_prompt | model | parser
merge_chain = merge_prompt | model | parser

# -------- Tool Definition --------
@tool
def generate_quiz(text: str, difficulty: str) :
    """
    Generates quiz questions based on difficulty level.
    Difficulty can be Easy, Medium, or Hard.
    """
    if difficulty == "Easy":
        return easy_quiz_chain.invoke({"text": text})
    elif difficulty == "Medium":
        return medium_quiz_chain.invoke({"text": text})
    else:
        return hard_quiz_chain.invoke({"text": text})


# -------- Agent Prompt --------
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that generates quizzes using tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

tools = [generate_quiz]

# -------- Create Agent --------
agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -------- Difficulty Selector --------
difficulty = st.selectbox(
    "Select Quiz Difficulty",
    ["Easy", "Medium", "Hard"]
)

# -------- PDF Upload --------

#Streamlit gives you a file-like object in memory, NOT a real file on disk.
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
#This creates a temporary file in your system. tempfile = built-in Python module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

#Reads entire uploaded PDF content as bytes.
#Writes those bytes into the temporary file.
        tmp_file.write(uploaded_file.read())

#This stores the file path as a string.
        temp_path = tmp_file.name

    try:
#PyPDFLoader requires a file path
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        full_text = "\n".join([doc.page_content for doc in docs])
        word_count = len(full_text.split())
        st.info(f"Total Words in PDF: {word_count}")


        st.success(f"PDF Loaded Successfully! Total Pages: {len(docs)}")

        if st.button("Generate Notes & Quiz"):
            with st.spinner("Generating content..."):

                # Generate Notes
                notes = notes_chain.invoke({"text": full_text})

                # Generate Quiz via Agent Tool
                quiz_response = agent_executor.invoke({
                    "input": f"Generate a {difficulty} quiz using the tool.",
                    "text": full_text,
                    "difficulty": difficulty
                })

                quiz = quiz_response["output"]

                # Merge Results
                result = merge_chain.invoke({
                    "notes": notes,
                    "quiz": quiz
                })

            st.markdown("# 📄 Generated Content")
            st.markdown(result)


  # -------- COST CALCULATOR --------

            # PROMPT_COST_PER_1K = 0.0002
            # COMPLETION_COST_PER_1K = 0.0004

            # cost = ((total_prompt / 1000) * PROMPT_COST_PER_1K) + \
            #        ((total_completion / 1000) * COMPLETION_COST_PER_1K)

            # st.markdown("## 💰 Estimated Cost")
            # st.write(f"Cost for this request: ${cost:.6f}")

    finally:
        os.remove(temp_path)
