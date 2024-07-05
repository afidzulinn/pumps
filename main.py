from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io
import uvicorn

app = FastAPI()

model = YOLO("models/pumps.pt")

# class_names = [f"pump {i}" for i in range (0, 101, 5)]

class_names = ["pump 0%", "pump 5%", "pump 10%", "pump 15%", "pump 20%", "pump 25%",
                "pump 30%", "pump 35%", "pump 40%", "pump 45%", "pump 50%", "pump 55%",
                "pump 60%", "pump 65%", "pump 70%", "pump 75%", "pump 80%", "pump 85%",
                "pump 90%", "pump 95%", "pump 100%"]

def detect(image, class_names):
    results = model.predict(image)[0]
    detection = {}
    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        class_name = class_names[class_id]
        percentage = class_name.split()[-1][:-1]
        detection = {
            # 'class_id': class_id,
            'class_name': class_name,
            'percentage': int(percentage)
        }
    return detection

@app.get("/")
async def check_status():
    return {"message": "Pump Classification API"}

@app.post("/detect")
async def upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    detections = detect(image, class_names)

    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    uvicorn.run(app)
