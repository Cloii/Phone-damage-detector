from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from groq import Groq
from PIL import Image
import io, base64, uvicorn
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
        conf=0.25,       # Lower threshold = catches more damage
        iou=0.45,        # Tighter overlap filtering
        augment=True,    # Test-time augmentation for better detection
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

    # Also encode original image for vision fallback in /report
    orig_buffer = io.BytesIO(image_bytes)
    orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode()

    return {
        "image": img_str,
        "original_image": orig_b64,
        "detections": detections
    }


@app.post("/report")
async def report(data: dict):
    detections = data.get("detections", [])
    original_image_b64 = data.get("original_image", None)

    damage_text = "\n".join(
        [f"- {d['label']} ({d['confidence']}% confidence)" for d in detections]
    ) if detections else "No damage flagged by object detection model."

    # Vision analysis of original image via Groq
    vision_analysis = ""
    if original_image_b64:
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
                                    "url": f"data:image/jpeg;base64,{original_image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": """You are an expert phone damage inspector. 
Carefully examine this phone image and identify ALL physical damage you can see, including but not limited to:

SCREEN DAMAGE:
- Cracked screen (spider cracks, corner cracks, full breaks)
- Shattered glass
- LCD bleed / dark spots / dead pixels
- Scratches on screen

BUTTON DAMAGE:
- Broken or missing home button (especially iPhone home buttons — look for cracks, missing button, sunken button)
- Damaged power/side button
- Broken volume buttons
- Stuck or recessed buttons

BODY/FRAME DAMAGE:
- Bent or warped frame
- Cracked back glass or plastic
- Dents or deep scratches on chassis
- Chipped corners or edges
- Loose or detached parts
- Water damage indicators (discoloration, corrosion marks)
- Missing or damaged camera lens cover

List EVERY damage you observe, even minor ones. Be aggressive — do not dismiss subtle damage. 
Format your response as a bulleted list of specific findings."""
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            vision_analysis = vision_response.choices[0].message.content
        except Exception as e:
            vision_analysis = f"(Vision analysis unavailable: {str(e)})"

    prompt = f"""You are a senior phone repair technician with 15+ years of experience repairing all brands (iPhone, Samsung, Xiaomi, Huawei, etc.).

OBJECT DETECTION RESULTS:
{damage_text}

VISUAL INSPECTION FINDINGS:
{vision_analysis if vision_analysis else "Not available."}

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

    return {"report": response.choices[0].message.content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)