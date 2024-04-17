import cv2
import os
import face_recognition

def load_image(image_path):
    """Load an image from the given file path."""
    return face_recognition.load_image_file(image_path)

def get_face_encodings(image):
    """Get face encodings from the given image."""
    return face_recognition.face_encodings(image)

def compare_faces(input_face_encodings, dataset_face_encodings, threshold=0.6):
    """Compare face encodings and return a match percentage."""
    match_percentage = 0
    for input_encoding in input_face_encodings:
        for dataset_encoding in dataset_face_encodings:
            match = face_recognition.compare_faces([input_encoding], dataset_encoding, tolerance=threshold)
            if match[0]:
                match_percentage = face_recognition.face_distance([input_encoding], dataset_encoding)[0]
                match_percentage = (1 - match_percentage) * 100  # Convert to percentage
                return match_percentage
    return match_percentage

def detect_faces(image):
    """Detect faces in the given image."""
    return face_recognition.face_encodings(image)

def capture_image():
    """Capture an image from the webcam."""
    cap = cv2.VideoCapture(0)  # Access the webcam
    ret, frame = cap.read()  # Capture a frame
    cap.release()  # Release the webcam
    return frame

def recognize_faces(input_image, dataset_folder  = "./Face_Recog/dataset", threshold=50):
    """Recognize faces in the input image against a dataset."""
    # Get face encodings from input image
    input_face_encodings = get_face_encodings(input_image)
    
    # Initialize results dictionary
    match_results = {}
    results = []
    
    # Iterate over images in the dataset folder
    for filename in os.listdir(dataset_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            print(filename)
            dataset_image_path = os.path.join(dataset_folder, filename)
            dataset_image = load_image(dataset_image_path)
            dataset_face_encodings = detect_faces(dataset_image)
            
            if len(input_face_encodings) == 0 or len(dataset_face_encodings) == 0:
                print("No face detected")
                match_results[filename] = 0, "No face detected"
                continue

            # Compare face encodings
            match_percentage = compare_faces(input_face_encodings, dataset_face_encodings)
            
            if match_percentage >= threshold:
                results.append({"image": filename, "match_percentage": match_percentage})
                match_results[filename] = match_percentage, "The faces are the same"
                # Save the matching image
                save_path = os.path.join('matches', filename)
                cv2.imwrite(save_path, input_image)
            else:
                results.append({"image": filename, "match_percentage": match_percentage})
                match_results[filename] = match_percentage, "The faces are different"
    max_percentage = 0
    res = ''
    for result in results:
        if result["match_percentage"]>max_percentage:
            max_percentage = result["match_percentage"]
            res = result["image"]

    return {"matched":res}

def main():
    dataset_folder = './Face_Recog/dataset'
    matching_threshold = 50

    # Capture image from webcam
    input_image = capture_image()

    # Recognize faces in the captured image
    os.makedirs('matches', exist_ok=True)
    results = recognize_faces(input_image, dataset_folder, threshold=matching_threshold)

if __name__ == "__main__":
    main()
