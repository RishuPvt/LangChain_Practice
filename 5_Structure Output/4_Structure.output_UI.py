import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# JSON Schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)


st.set_page_config(page_title="Review Analyzer")

st.title("Product Review Analyzer")
st.write("Paste a product review below and get structured insights instantly.")

review_text = st.text_area(
    "Enter Review Text",
    height=250,
    placeholder="Paste the full review here..."
)

if st.button("Analyze Review"):
    if not review_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                result = structured_model.invoke(review_text)

                st.success("Analysis Complete ")

                st.subheader(" Structured Output")
#st.json() renders JSON data in a clean, interactive, pretty-formatted way in the Streamlit UI.
                st.json(result)

    
                st.subheader(" Summary")
                st.write(result.get("summary"))

                st.subheader(" Sentiment")
                st.write(result.get("sentiment"))

                st.subheader(" Key Themes")
                st.write(result.get("key_themes"))

                if result.get("pros"):
                    st.subheader("Pros")
                    st.write(result.get("pros"))

                if result.get("cons"):
                    st.subheader(" Cons")
                    st.write(result.get("cons"))

                if result.get("name"):
                    st.subheader("Reviewer")
                    st.write(result.get("name"))

            except Exception as e:
                st.error(f"Error: {e}")
