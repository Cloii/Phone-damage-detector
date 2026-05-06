from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import io, base64, uvicorn, json, asyncio
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

model = YOLO("best.pt")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup: force YOLO through a full inference pass on startup
    # so the first real /detect request is never slow
    print("Warming up YOLO model...")
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    model(dummy, verbose=False)
    print("Model is hot and ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def resize_image(image: Image.Image, max_size=640) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def run_yolo(image: Image.Image):
    # Fix: convert RGBA/PNG to RGB before YOLO
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    results = model(
        image,
        conf=0.25,
        iou=0.45,
        augment=False,
        agnostic_nms=True,
        imgsz=640
    )

    result_img = results[0].plot()
    pil_img = Image.fromarray(result_img)

    # Fix: ensure result image is also RGB before saving as JPEG
    if pil_img.mode in ("RGBA", "P", "LA"):
        pil_img = pil_img.convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=75)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        detections.append({
            "label": label,
            "confidence": round(confidence * 100, 1)
        })

    return img_str, detections


def _groq_vision_sync(orig_b64: str):
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
                            "text": """You are an expert phone damage inspector with strict assessment standards.

Examine this phone image carefully and identify ALL physical damage.

CRITICAL RULES — READ BEFORE ANALYZING:
- "Cracked Screen" means ONLY visible glass cracks, fracture lines, spider cracks, or shattered glass on the screen surface. Do NOT use this label for display issues.
- "LCD Bleed" means color bleeding, dark patches, or light leaking from edges.
- "Screen Distortion" means lines, color artifacts, flickering, or abnormal display output — this is INTERNAL display damage, NOT a crack.
- "Dead Pixels" means black/white stuck pixels.
- NEVER label display artifacts, color lines, or distortion as "Cracked Screen" — those are separate conditions.
- Only report "Cracked Screen" if you can clearly see fracture lines on the glass itself.
- If you are not 100% sure glass is cracked, do NOT include it. Use "Screen Distortion" or "LCD Damage" instead.
- Do NOT inflate confidence scores. If you are uncertain, lower the confidence value.

Respond ONLY with a JSON array. No explanation, no markdown, just raw JSON.
Each item must have:
- "label": precise damage name — be specific (e.g. "LCD Bleed", "Vertical Screen Distortion", "Cracked Glass", "Bent Frame")
- "confidence": integer 0-100 — be honest, do not inflate scores
- "location": where on the phone (e.g. "center screen", "top-left corner", "left edge")

Also check for:
- Broken/missing/sunken home button (iPhones especially)
- Damaged power or volume buttons
- Bent or warped frame, dents, chipped corners
- Cracked back glass or plastic
- Water damage discoloration or corrosion marks
- Damaged camera lens cover
- Loose or detached parts

If no damage found, return: []"""
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        raw = vision_response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return []


async def run_vision(orig_b64: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _groq_vision_sync, orig_b64)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    resized = resize_image(image, max_size=640)

    # Fix: convert RGBA/PNG to RGB before saving as JPEG
    if resized.mode in ("RGBA", "P", "LA"):
        resized = resized.convert("RGB")

    orig_buffer = io.BytesIO()
    resized.save(orig_buffer, format="JPEG", quality=80)
    orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode()

    # Run YOLO and Vision in parallel
    loop = asyncio.get_event_loop()
    yolo_future = loop.run_in_executor(None, run_yolo, resized)
    vision_future = run_vision(orig_b64)

    try:
        (img_str, detections), additional_damage = await asyncio.gather(
            yolo_future, vision_future
        )
    except Exception:
        # Fallback if parallel execution fails
        img_str, detections = run_yolo(resized)
        additional_damage = []

    return {
        "image": img_str,
        "original_image": orig_b64,
        "detections": detections,
        "additional_damage": additional_damage
    }


@app.post("/report")
async def report(data: dict):
    detections = data.get("detections", [])
    additional_damage = data.get("additional_damage", [])

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

IMPORTANT: Only include damages that are clearly supported by the findings above.
Do NOT invent or assume damage that was not detected. If a damage type was not found, do not mention it.

Based on the confirmed findings, provide a comprehensive damage report:

1. **Complete Damage Summary** — list only confirmed damage with location and severity

2. **Severity Level** — Overall: Minor / Moderate / Severe / Critical
   - Rate each confirmed damage individually

3. **Repair Priority** — What needs fixing immediately vs. can wait

4. **Step-by-Step Repair Recommendations** — specific to each confirmed damage

5. **Estimated Repair Cost Range** (PHP and USD) — per damage type and total

6. **Usability Assessment** — Is the phone safe to use? What functions are affected?

7. **Data Risk** — Is there any risk of data loss?

Be accurate and honest. Do not exaggerate or add damage that was not detected."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200
        )
    )

    return {
        "report": response.choices[0].message.content,
        "vision_findings": vision_text
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)