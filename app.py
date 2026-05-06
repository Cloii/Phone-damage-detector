from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import io, base64, uvicorn, os, json
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
model       = YOLO("best.pt")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── /detect ──────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image       = Image.open(io.BytesIO(image_bytes))

    # 1. YOLO — screen crack specialist
    results    = model(image)
    result_img = results[0].plot()
    pil_img    = Image.fromarray(result_img)
    buffer     = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    yolo_detections = []
    for box in results[0].boxes:
        label      = model.names[int(box.cls)]
        confidence = float(box.conf)
        yolo_detections.append({
            "label":      label,
            "confidence": round(confidence * 100, 1)
        })

    # 2. Groq Vision — detects everything else
    image_b64         = base64.b64encode(image_bytes).decode()
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
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """You are a professional phone damage inspector with expert knowledge of display and hardware faults.

Carefully examine this phone image for ALL visible damage.

== DISPLAY DAMAGE — look hard for these (HIGH PRIORITY) ==
- Vertical or horizontal colored lines (green, pink, yellow, white, black lines across screen)
- Screen discoloration or color bleeding (areas with wrong colors, rainbow effects)
- LCD damage or internal display damage (bright patches, dark patches, washed out areas)
- Burn-in or ghost images (faint permanent images burned into screen)
- Flickering patterns or distorted display content
- Black spots or dark blotches on screen
- White or bright patches (backlight bleed)
- Partial screen blackout (section of screen not displaying correctly)
- Pixelation or blurry zones on display
- Rainbow or iridescent patterns on screen

== PHYSICAL DAMAGE — also check ==
- Back glass cracks or scratches
- Body scratches on frame or sides
- Dents or physical deformation
- Bent or warped frame
- Camera lens scratches or cracks
- Charging port or button damage
- Missing pieces or chips

== DETECTION RULES ==
- COLORED LINES on screen = always report as "Vertical/Horizontal display lines (LCD damage)" with 95%+ confidence
- ANY unusual colors or patterns on screen that don't look like normal app UI = report as display damage
- Be AGGRESSIVE detecting display anomalies — this is the most important damage to catch
- Only skip if the area is completely undamaged
- Confidence scale:
  90-100% = unmistakably clear damage
  70-89%  = clearly visible damage
  50-69%  = likely damage, possible lighting issue
  Below 50% = do NOT report

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "additional_damage": [
    {"label": "damage type", "confidence": 95, "location": "where on phone"}
  ]
}
If truly no damage visible: {"additional_damage": []}"""
                        }
                    ]
                }
            ],
            temperature=0.1
        )

        vision_text = vision_response.choices[0].message.content.strip()

        # Strip markdown if model wraps response
        if "```" in vision_text:
            vision_text = vision_text.split("```")[1]
            if vision_text.startswith("json"):
                vision_text = vision_text[4:]
        vision_text = vision_text.strip()

        vision_data       = json.loads(vision_text)
        additional_damage = vision_data.get("additional_damage", [])

        # Filter low confidence
        additional_damage = [d for d in additional_damage if d.get("confidence", 0) >= 50]

        # Remove duplicates with YOLO
        additional_damage = [
            d for d in additional_damage
            if "screen crack" not in d["label"].lower()
            and "cracked screen" not in d["label"].lower()
        ]

    except Exception as e:
        print(f"[Vision Error] {e}")
        additional_damage = []

    return {
        "image":             img_str,
        "detections":        yolo_detections,
        "additional_damage": additional_damage
    }


# ── /report ──────────────────────────────────────────────────────────────────
@app.post("/report")
async def report(data: dict):
    detections = data.get("detections",        [])
    additional = data.get("additional_damage", [])

    if not detections and not additional:
        return {"report": "No damage detected. Your phone appears to be in good condition."}

    yolo_text = "\n".join(
        [f"- {d['label']} ({d['confidence']}% confidence)" for d in detections]
    ) if detections else "- None detected"

    ai_text = "\n".join(
        [f"- {d['label']} at {d.get('location', 'unspecified')} ({d['confidence']}% confidence)"
         for d in additional]
    ) if additional else "- None detected"

    prompt = f"""You are a professional phone repair technician writing a damage report for a customer.

YOLO Model Detected (screen crack specialist):
{yolo_text}

AI Vision Analysis (full damage scan):
{ai_text}

Write a clear, professional repair report with these sections:

1. DAMAGE OVERVIEW
   Summarize all damage found. For display damage like colored lines or LCD issues,
   explain what it means in plain language (e.g. "The vertical colored lines indicate
   internal display damage, likely a damaged LCD ribbon cable or broken display panel").

2. SEVERITY
   Overall rating: Minor / Moderate / Severe — explain why.

3. REPAIR PRIORITY
   List repairs in order of urgency with what happens if left unrepaired.

4. ESTIMATED COST (PHP)
   Realistic cost range per repair and a total estimate.
   Use current market rates for phone repairs.

5. USABILITY
   Can the phone still be used safely? Any risks to the user?

6. RECOMMENDATION
   Repair, replace, or monitor? Be direct and practical.

Keep it honest, clear, and helpful."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"report": response.choices[0].message.content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)