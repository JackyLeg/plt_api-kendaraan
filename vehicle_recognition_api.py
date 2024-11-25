from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from transformers import pipeline

# Inisialisasi FastAPI
app = FastAPI()

# Inisialisasi model image recognition
# Gunakan model yang mendukung pengenalan kendaraan (misalnya, "google/vit-base-patch16-224")
model = pipeline("image-classification", model="google/vit-base-patch16-224")

@app.post("/recognize_vehicle")
async def recognize_vehicle(file: UploadFile = File(...)):
    try:
        # Baca file gambar
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Lakukan prediksi
        predictions = model(image)

        # Filter hanya untuk kendaraan
        vehicle_classes = ["car", "truck", "bus", "bicycle", "motorcycle", "van"]
        vehicle_predictions = [
            {"label": pred["label"], "score": pred["score"]}
            for pred in predictions
            if any(vehicle in pred["label"].lower() for vehicle in vehicle_classes)
        ]

        if not vehicle_predictions:
            return JSONResponse(content={"message": "Tidak ada kendaraan yang terdeteksi."}, status_code=404)

        return {"message": "Kendaraan terdeteksi.", "predictions": vehicle_predictions}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint test sederhana
@app.get("/")
def root():
    return {"message": "Vehicle Recognition API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
