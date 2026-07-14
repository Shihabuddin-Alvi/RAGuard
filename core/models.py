import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_model(user_prompt, system_prompt):
    response = CLIENT.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text
