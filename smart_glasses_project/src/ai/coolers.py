import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_commentary(scene_description):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for visually impaired people."},
                {"role": "user", "content": f"A person is walking and sees: {scene_description}. Describe the scene for a visually impaired person."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ GPT error:", e)
        return None
