from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)



# from langchain_text_splitters import CharacterTextSplitter

# text = """LangChain is a powerful framework for developing applications powered by language models.
# It enables developers to chain together components like LLMs, prompts, and memory to create advanced conversational AI systems.
# Text splitters in LangChain help break large documents into smaller pieces for processing."""

# splitter = CharacterTextSplitter(
#     chunk_size=40,
#     chunk_overlap=10,
#     separator=" "
# )

# chunks = splitter.split_text(text.replace("\n", " "))

# print("📄 Number of Chunks:", len(chunks))
# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:\n{chunk}")