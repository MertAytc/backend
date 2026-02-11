from dotenv import load_dotenv
import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol et.")

client = OpenAI(api_key=api_key)

app = FastAPI()


class QuizAnswers(BaseModel):
    answers: list[str]


@app.post("/recommend-car")
async def recommend_car(data: QuizAnswers):
    try:
        # 1. Flutter'dan gelen cevapları birleştirip bir prompt oluşturuyoruz
        user_answers_text = ", ".join(data.answers)

        system_instruction = """
        Sen bir araba uzmanısın. Kullanıcının verdiği cevaplara göre ona en uygun arabayı öner.
        Cevabını SADECE aşağıdaki JSON formatında ver, başka hiçbir metin ekleme:
        {
            "car": "Marka Model",
            "comment": "Neden bu aracı önerdiğine dair kısa, eğlenceli bir yorum.",
            "image": "Aracın temsili bir görsel URL'si (wikimedia veya halka açık bir url)"
        }
        """

        prompt = f"Kullanıcının test cevapları: {user_answers_text}. Buna göre bir araba öner."

        # 2. OpenAI'a istek atıyoruz (Doğru Metot)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}  # JSON dönmesi için zorluyoruz
        )

        # 3. Gelen cevabı alıyoruz
        content = response.choices[0].message.content

        # 4. JSON'a çevirip Flutter'a gönderiyoruz
        result = json.loads(content)
        return result

    except Exception as e:
        # Hata olursa loglayıp basit bir hata mesajı dönüyoruz
        print(f"Hata oluştu: {e}")
        return {
            "car": "Hata",
            "comment": "Bir şeyler ters gitti, ama Toyota Corolla her zaman güvenli bir limandır.",
            "image": "https://upload.wikimedia.org/wikipedia/commons/5/5e/2019_Toyota_Corolla.jpg"
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # 'main:app' kısmındaki 'main' dosya adınla aynı olmalı (örn: main.py)
    uvicorn.run(app, host="0.0.0.0", port=port)