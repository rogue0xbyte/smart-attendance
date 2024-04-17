from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import os
import haarcascade_facecount
from Face_Recog import facerecognition

app = FastAPI()

def save_file(contents, file_path):
    with open(file_path, "wb") as f:
        f.write(contents)
def capture_image():
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)  # Access the webcam
    ret, frame = cap.read()  # Capture a frame
    cap.release()  # Release the webcam
    return frame

@app.post("/facecount")
async def facecount(file: UploadFile = File(...)):
    # Define file path to save uploaded image
    file_path = "uploaded_image.jpg"
    
    # Save the uploaded file
    contents = await file.read()
    save_file(contents, file_path)
    
    # Pass the image to haarcascade_facecount
    result = haarcascade_facecount.main(file_path)
    face_count = result[0]
    result_image = result[1]
    
    # Return face count and image path
    return {"face_count": face_count, "image_path": result_image}

@app.post("/recognise")
async def recognise(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    contents = await file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform face recognition
    result = facerecognition.recognize_faces(input_image=image)
    
    # Print result
    print(result)
    
    # Return JSON response
    return JSONResponse(content=result)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()