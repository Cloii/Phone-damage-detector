from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import io, base64, uvicorn, json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(
        image,
        conf=0.25,
        iou=0.45,
        augment=True,
        agnostic_nms=True
    )

    result_img = results[0].plot()
    pil_img = Image.fromarray(result_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        detections.append({
            "label": label,
            "confidence": round(confidence * 100, 1)
        })

    orig_b64 = base64.b64encode(image_bytes).decode()

    # Run vision analysis immediately at /detect so frontend gets it right away
    additional_damage = []
    try:
        vision_response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{orig_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """You are an expert phone damage inspector.
Carefully examine this phone image and identify ALL physical damage you can see.

Respond ONLY with a JSON array. No explanation, no markdown, just raw JSON.
Each item must have:
- "label": short damage name (e.g. "Cracked Screen", "Broken Home Button", "Bent Frame")
- "confidence": integer 0-100 representing how certain you are
- "location": where on the phone (e.g. "top-left corner", "home button area", "back glass")

Include ALL damage found, even minor. Be aggressive — do not dismiss subtle damage.
Check specifically for:
- Cracked/shattered screen, LCD bleed, dead pixels, scratches
- Broken/missing/sunken home button (iPhones especially)
- Damaged power button, volume buttons
- Bent or warped frame, dents, chipped corners
- Cracked back glass or plastic
- Water damage discoloration or corrosion
- Damaged camera lens cover
- Loose or detached parts

If no damage found, return an empty array: []

Example format:
[
  {"label": "Cracked Screen", "confidence": 92, "location": "top-right corner"},
  {"label": "Broken Home Button", "confidence": 78, "location": "bottom center"},
  {"label": "Chipped Frame", "confidence": 65, "location": "left edge"}
]"""
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        raw = vision_response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        additional_damage = json.loads(raw)
    except Exception as e:
        additional_damage = []

    return {
        "image": img_str,
        "original_image": orig_b64,
        "detections": detections,
        "additional_damage": additional_damage  # ← matches frontend exactly
    }


@app.post("/report")
async def report(data: dict):
    detections = data.get("detections", [])
    additional_damage = data.get("additional_damage", [])
    original_image_b64 = data.get("original_image", None)

    damage_text = "\n".join(
        [f"- {d['label']} ({d['confidence']}% confidence)" for d in detections]
    ) if detections else "No damage flagged by YOLO model."

    vision_text = "\n".join(
        [f"- {d['label']} at {d.get('location', 'unknown')} ({d['confidence']}% confidence)"
         for d in additional_damage]
    ) if additional_damage else "No additional damage found by vision model."

    prompt = f"""You are a senior phone repair technician with 15+ years of experience repairing all brands (iPhone, Samsung, Xiaomi, Huawei, etc.).

YOLO OBJECT DETECTION RESULTS:
{damage_text}

AI VISION INSPECTION FINDINGS:
{vision_text}

Based on ALL the above findings, provide a comprehensive damage report:

1. **Complete Damage Summary** — list every identified damage, including physical component damage (home button, power button, volume buttons, charging port area, camera glass, frame, back cover)

2. **Severity Level** — Overall: Minor / Moderate / Severe / Critical
   - Also rate each damage individually

3. **Repair Priority** — What needs fixing immediately vs. can wait

4. **Step-by-Step Repair Recommendations** — specific to each damage found

5. **Estimated Repair Cost Range** (PHP and USD) — per damage type and total

6. **Usability Assessment** — Is the phone safe to use? What functions are affected?

7. **Data Risk** — Is there any risk of data loss?

Be thorough, specific, and do NOT downplay any damage found."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    return {
        "report": response.choices[0].message.content,
        "vision_findings": vision_text
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)