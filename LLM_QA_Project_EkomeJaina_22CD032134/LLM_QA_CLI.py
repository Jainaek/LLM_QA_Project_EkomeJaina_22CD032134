import os
import re
import string
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    exit()

genai.configure(api_key=API_KEY)

def basic_preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

def get_answer_from_llm(question: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def main():
    print("--- LLM Q&A Command-Line Interface ---")
    user_question = input("\nEnter your question: ")

    if not user_question.strip():
        print("No question entered. Exiting.")
        return

    processed_q = basic_preprocess(user_question)

    print(f"\nOriginal Question: {user_question}")
    print(f"Preprocessed Text: {processed_q}")

    print("\nSending question to LLM...")
    llm_answer = get_answer_from_llm(user_question)

    print("\n==================================")
    print("🤖 Final LLM Answer:")
    print(llm_answer)
    print("==================================")

if __name__ == "__main__":
    main()
