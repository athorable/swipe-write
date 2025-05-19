from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json
import os
import base64

# Load .env file
load_dotenv()

# Create OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI()

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Utility: Scrape and extract text from a web page
def get_page_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return cleaned[:5000]
    except Exception as e:
        return f"Could not fetch content from {url}. Error: {e}"

# Utility: Summarize text using GPT
def summarize_text(text):
    try:
        summary_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following website content in 5 bullet points with key insights or tips."},
                {"role": "user", "content": text}
            ]
        )
        return summary_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable. Error: {e}"

# Request model for basic chat
class ChatRequest(BaseModel):
    message: str

# Homepage (chat UI)
@app.get("/", response_class=HTMLResponse)
async def serve_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Text-only chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Swipe Write, a sassy, razor-sharp, no-BS dating guru who delivers hilariously blunt, swagger-filled advice that transforms boring dating profiles and limp messages into irresistible, high-flirt masterpieces. You balance flirty with fun, sarcastic with charming, and never cross into creepy or mean. Your tone is bold, teasing, and confident—with a flair for dramatic transformations and mic-drop moments."},
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image + message upload endpoint
@app.post("/analyze")
async def analyze_image(message: str = Form(...), image: UploadFile = File(None)):
    try:
        image_response_text = None

        if image:
            contents = await image.read()
            base64_image = base64.b64encode(contents).decode("utf-8")
            image_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You're Swipe Write, the brutally fabulous dating coach who roasts photos with care and confidence. You point out red flags, good lighting, outfit wins, pose fails, and give real, high-flirt feedback that gets results—always cheeky, never mean."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            image_response_text = image_response.choices[0].message.content.strip()
            return {"image_response": image_response_text}

        # Fallback to chat if no image
        fallback_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Swipe Write, the same spicy dating profile guru..."},
                {"role": "user", "content": message}
            ]
        )
        return {"response": fallback_response.choices[0].message.content.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})