import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import tiktoken

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(page_title="Notes & Quiz Generator")
st.title("Notes & Quiz Generator (PDF Upload)")
st.write("Upload a PDF and generate **short notes** and **quiz questions**")

# ---------------- TOKENIZER ----------------

#cl100k_base tokenizer used by models 
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# ---------------- MODEL ----------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ---------------- PROMPTS ----------------
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

# ---------------- TOOL ----------------
@tool
def generate_quiz(text: str, difficulty: str) -> str:
    """
    Generate a quiz based on provided text and difficulty level.
    Difficulty can be Easy, Medium, or Hard.
    """
    if difficulty == "Easy":
        response = (easy_quiz_prompt | model).invoke({"text": text})
    elif difficulty == "Medium":
        response = (medium_quiz_prompt | model).invoke({"text": text})
    else:
        response = (hard_quiz_prompt | model).invoke({"text": text})

    return response.content

# ---------------- AGENT ----------------
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that generates quizzes using tools."),
        ("human", "{input}"),
#"This is a dynamic section that LangChain will automatically fill during runtime."
        ("placeholder", "{agent_scratchpad}")
    ]
)

tools = [generate_quiz]

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

# ---------------- UI ----------------
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
        token_count = count_tokens(full_text)

        st.success(f"PDF Loaded Successfully! Pages: {len(docs)}")
        st.info(f"Total Words: {word_count}")
        st.info(f"Exact Token Count (Prompt): {token_count}")

        # -------- TOKEN METER --------
        MAX_CONTEXT = 8000
        usage_ratio = min(token_count / MAX_CONTEXT, 1.0)

        st.progress(usage_ratio)
        st.write(f"Context Usage: {token_count}/{MAX_CONTEXT} tokens")

        if token_count > 7500:
            st.error(" Warning: Near model context limit!")

        if st.button("Generate Notes & Quiz"):
            with st.spinner("Generating content..."):

                # -------- NOTES --------
                 # Generate Notes
                notes_response = (notes_prompt | model).invoke({"text": full_text})
                notes = notes_response.content

                notes_usage = notes_response.response_metadata.get("token_usage", {})

                # -------- QUIZ --------
                 # Generate Quiz via Agent Tool
                quiz_response = agent_executor.invoke({
                    "input": f"Generate a {difficulty} quiz using the tool.",
                    "text": full_text,
                    "difficulty": difficulty
                })

                quiz = quiz_response["output"]

                # -------- MERGE --------
                merge_response = (merge_prompt | model).invoke({
                    "notes": notes,
                    "quiz": quiz
                })

                result = merge_response.content
                merge_usage = merge_response.response_metadata.get("token_usage", {})

            # -------- DISPLAY OUTPUT --------
            st.markdown("# 📄 Generated Content")
            st.markdown(result)

            # -------- TOKEN USAGE DISPLAY --------
            st.markdown("##  Actual LLM Usage")

            total_prompt = (
                notes_usage.get("prompt_tokens", 0)
                + merge_usage.get("prompt_tokens", 0)
            )

            total_completion = (
                notes_usage.get("completion_tokens", 0)
                + merge_usage.get("completion_tokens", 0)
            )

            total_tokens = total_prompt + total_completion

            st.write("Prompt Tokens Used:", total_prompt)
            st.write("Completion Tokens Used:", total_completion)
            st.write("Total Tokens Used:", total_tokens)



    finally:
        os.remove(temp_path)
