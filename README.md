# Smart Attendance Tracking

Welcome to Smart Attendance Tracking for Educational Institutions. This project is designed to address two key objectives: automating class attendance and mitigating proxy attendance.

**View Video Demo at <https://tiny.cc/smart-attendance> [https://drive.google.com/file/d/12n9oVFhOiYOGq-eI95tQk5Ei4-c0HHvx/view?usp=sharing]**

**Final Report and Plagiarism Report can be found in the `documentation` folder**

## Approach

For automating attendance, we create portals for Teachers and Students. Students' attendance is recorded via selfie images compared with dataset images (ID card images) stored in the database. To prevent proxy attendance, teachers take a group photo of the class, and an ML model performs HeadCount to limit the number of recorded selfies. This system utilizes Python 3.10.10 along with various packages such as OpenCV and face_recognition for face detection and recognition tasks.


## Conceptual Architecture

<img src="https://i.imgur.com/IH1DR1d.jpeg"/>

### Overview

#### Uploading Dataset for a Class:
- Students upload ID card images via a separate portal.
- Faces and OCR details (student name, roll number) are extracted and stored in the database.

#### Class Schedule in Teacher’s Login:
- Create Class: Teachers provide Course_ID, Teacher_ID, and Student IDs to create a Class.
- View Taken Class: Teachers can view the class schedule in tabular format.
- View Class: Shows Course Information (Course ID, Teacher ID) along with student attendance.

#### Taking Attendance for a Class:
1. Teacher logs into the Teacher Login and takes a group photo of the class for HeadCount.
2. Students log in and take selfies for attendance, which are compared with the ID card dataset faces.
3. Attendance is sent to the Teacher’s Login for the class.

### User Flow

#### Professor/Teacher:
1. Registration: Sign-up through the `/register` endpoint, redirected to the login page upon successful registration.
2. Login: Access via ID and password.
3. Dashboard: Provides links to:
   - Create a Class: `/create_class` endpoint to insert a new class into the database.
   - View Classes: Lists available classes.
   - Initialize Attendance: Generates a unique ID (UID) for attendance tracking; UID is distributed to students for marking attendance.

#### Students:
1. Registration: Required before participating in class or marking attendance.
2. Attendance Marking: Via a link containing UID, students upload a selfie and use their ID card for verification.
3. ID Verification: Utilizes PyTesseract trained on the MNIST dataset to recognize and extract text from student ID cards, mapping them to database entries (roll number and full name).

### Flow of Programs

#### Backend:
- `App.py`: Main FastAPI application managing backend operations.
- `FaceCounter.py`: Script responsible for counting faces using a cascade classifier.
- `Models/TypeA` and `TypeB`: Jupyter notebooks detailing face counting models for different configurations.
- `FaceRecog.py`: Script for face recognition, matching individuals against a predefined dataset.
- `/FaceCounting`: Directory containing algorithms and utilities for face counting functionality.
- `TestJPGs`: Directory with sample images for testing face recognition and counting models.
- `Dockerfile`: Configuration for containerizing the backend environment using Docker.


#### Head Count Algorithm

The method of face detection in pictures is very complicated because of variability it presents in human faces such as position and orientation, skin colour, pose, expression,  the presence of glasses or facial hair, differences in camera gain, lighting conditions, and image resolution.

Object detection is one of the Computer Technologies which is connected to the Computer Vision and Image Processing. It interacts with detecting instances of an object such as Human Faces, Building, Tree, Car, etc. The primary aim of face detection algorithms is to determine whether there is any face in an Image or not. 

We are provided with a training set of images with coordinates of bounding box and head count for each image and need to predict the headcount for each image in the test set.

The evaluation metric is RMSE (root mean squared error) over the head counts predicted for test images.

1. **Input**: The system expects input images in high quality to ensure accurate face detection and recognition. Ensure that input images are not heavily compressed to maintain quality.
   
2. **Face Detection**: The main program, `haarcascade_facecount.py`, utilizes the Haar cascade classifier for face detection. This classifier is loaded from the `haarcascade_frontalface_default.xml` file.

3. **Head Count**: After detecting faces in the input image, the system prints the number of faces found.

4. **Face Recognition**: The system then initiates the face recognition model to recognize the detected faces. This step relies on the `facerecognition` module.

5. **Output**: Rectangles are drawn around the detected faces in the original image, and the resulting image with rectangles is saved to the file system as `saved.jpg`.

The main program consists of the following functions:

- **`main()`**: This function is the entry point of the program. It loads the input image, detects faces using the Haar cascade classifier, prints the number of faces found, draws rectangles around the detected faces, saves the resulting image with rectangles, and initiates the face recognition process.
  
- **`load_face_recognition_model()`**: This function loads the pre-trained face recognition model required for recognizing faces in the input image.

The `haarcascade_frontalface_default.xml` file is a cascading classifier used for face detection. It contains a set of pre-trained features and weights that enable the system to efficiently detect human faces in images. The XML file structure includes specifications of classifier types, stages, decision trees, and feature descriptors, which are essential for the accurate detection of frontal faces.

#### Frontend:
- `App.py`: Primary FastAPI application for the frontend, managing user interactions and visual interfaces.
- `Requirements.txt`: List of necessary libraries and dependencies.
- `Dockerfile`: Script for Dockerizing the frontend.
- `GoogleOSIR.py` and `VisionAPI.json`: Scripts and configuration files for integrating Google Vision API.
- `/Static/Uploads`: Folder for storing user-uploaded files.
- `/Templates`: Collection of HTML templates for various interfaces.
- `Docker-compose.yml`: YAML file detailing Docker services.
- `Dump-face-app.sql`: SQL dump file containing database schemas and tables.

## Deployment with Docker

To deploy the Smart Attendance Tracking system using Docker, follow these steps:

1. Ensure Docker and Docker Compose are installed on your system.
2. Navigate to the root directory containing the `docker-compose.yml` file and two folders with Dockerfiles for backend and frontend.
3. Run the following command to build and start the containers: `docker-compose up --build`
4. Once the containers are running, access the system via your web browser using the specified URL.

By following these instructions, you can quickly deploy the Smart Attendance Tracking system in your educational institution, streamlining attendance management and enhancing accountability.

## Contributors

- 21Z201 - Aadil Arsh S R
- 21Z202 - Aaditya Rengarajan
- 21Z217 - Gaurav Vishnu N
- 21Z218 - Hareesh S
- 21Z247 - S Karun Vikhash
- 21Z248 - Sanjay Kumaar Eswaran