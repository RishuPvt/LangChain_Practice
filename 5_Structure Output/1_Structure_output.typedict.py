from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

class Review(TypedDict):
    summary: str
    sentiment: str
    pros: str
    cons: str


structured_llm = gemini.with_structured_output(Review)

text_to_analyze = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
"""

result = structured_llm.invoke(text_to_analyze)

#print(result)
print(f"\nSummary: {result['summary']}")
print(f"Sentiment: {result['sentiment']}")