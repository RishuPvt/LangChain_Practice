#LangChain  removed StructuredOutputParser in newer versions.
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]
parser= StructuredOutputParser.from_response_schemas(schema)

template =PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'topic':'delhi'})

print(result)