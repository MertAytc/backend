import os
import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class QuizAnswers(BaseModel):
    answers: list[str]


@app.get("/")
def health_check():
    return {"status": "API"}
    
@app.post("/recommend-car")
async def recommend_car(data: QuizAnswers):
    try:
        user_answers_text = ", ".join(data.answers)

        system_instruction = """
        SADECE JSON :
        {
            "car": "...",
            "comment": "...",
            "image": "https://..."
        }
        """

        prompt = f"Cevaplar: {user_answers_text}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print("HATA:", e)
        return {
            "car": "Toyota Corolla",
            "comment": "AI çalışmadı ",
            "image": "https://www.sixt.com.tr/storage/cache/3efbc805ae81badb5b348158fd337faed4ab35f5.webp"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


