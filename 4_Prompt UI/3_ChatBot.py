from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-3-flash-preview")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Append user input
    chat_history.append(f"You: {user_input}")

    # Tell it to remember previous conversation explicitly
    prompt = "You are a helpful assistant. Continue the conversation based on previous context."
    prompt += "\n".join(chat_history) 


    result = model.invoke(prompt)
    print("AI: ", result)

    chat_history.append(f"AI: {result}")


print(chat_history)
