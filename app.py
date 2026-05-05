from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import io, base64, uvicorn
import os
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    results = model(image)
    result_img = results[0].plot()
    pil_img = Image.fromarray(result_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        detections.append({"label": label, "confidence": round(confidence * 100, 1)})
    return {"image": img_str, "detections": detections}

@app.post("/report")
async def report(data: dict):
    detections = data.get("detections", [])
    if not detections:
        return {"report": "No damage detected. Your phone appears to be in good condition."}

    damage_text = "\n".join(
        [f"- {d['label']} ({d['confidence']}% confidence)" for d in detections]
    )

    prompt = f"""You are a phone repair expert. A phone was scanned and the following damage was detected:

{damage_text}

Please provide:
1. A brief damage assessment
2. Severity level (Minor / Moderate / Severe)
3. Recommended repair steps
4. Estimated repair cost range
5. Whether the phone is still usable as-is

Be concise and practical."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"report": response.choices[0].message.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
