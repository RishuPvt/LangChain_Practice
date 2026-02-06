#RunnableSequance is Sequantial chain of runnables in langchain that execute each step one after other , passing the output of one step as the input to the next.

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser = StrOutputParser()

prompt = PromptTemplate(
   template='Write a joke about {topic}',
    input_variables=['topic'],
  
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))

