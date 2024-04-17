from fastapi import *
from fastapi.responses import HTMLResponse
import cv2
import dlib
import os
# import pytesseract
from modules.google_ocr import read_image
import re
import psycopg2
from psycopg2 import sql
from PIL import Image
from fastapi.templating import Jinja2Templates
from typing import List
import json
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")

new_path = "uploads"
detector = dlib.get_frontal_face_detector()

# PostgreSQL database configuration
DATABASE_URL = 'postgresql://postgres:inr_db@db.inr.intellx.in/faceapi'
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

def create_table():
    """Create the 'id_info' table if it doesn't exist."""
    queries = ["""
    CREATE TABLE IF NOT EXISTS id_info (
        id SERIAL PRIMARY KEY,
        name TEXT,
        output TEXT,
        image_data BYTEA
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS classes (
        course_id VARCHAR(10) PRIMARY KEY,
        teacher_id SERIAL,
        student_ids JSONB
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id SERIAL PRIMARY KEY,
        u_id INT,
        roll_number VARCHAR(255)
    )
    """
    ]
    for query in queries:
        cursor.execute(query)
    conn.commit()

create_table()

def execute_sql(sql_command, values=None, type=None):
    with conn.cursor() as cursor:
        cursor.execute(sql_command, values)
        if (str(sql_command).strip().split(" ")[0].upper() in ['SELECT', 'SHOW']) or (type=="SEL"):
            return cursor.fetchall()
        else:
            conn.commit()
            return {"message": "Operation successful"}

def MyRec(rgb, x, y, w, h, v=20, color=(200, 0, 0), thickness=2):
    """Draw stylish rectangles around the objects with optional corners."""
    cv2.rectangle(rgb, (x, y), (x + w, y + h), color, thickness)
    # Drawing corner lines to add styling
    # Top-left corner
    cv2.line(rgb, (x, y), (x + v, y), color, thickness)
    cv2.line(rgb, (x, y), (x, y + v), color, thickness)
    # Top-right corner
    cv2.line(rgb, (x + w, y), (x + w - v, y), color, thickness)
    cv2.line(rgb, (x + w, y), (x + w, y + v), color, thickness)
    # Bottom-left corner
    cv2.line(rgb, (x, y + h), (x, y + h - v), color, thickness)
    cv2.line(rgb, (x, y + h), (x + v, y + h), color, thickness)
    # Bottom-right corner
    cv2.line(rgb, (x + w, y + h), (x + w, y + h - v), color, thickness)
    cv2.line(rgb, (x + w, y + h), (x + w - v, y + h), color, thickness)

def save(img, name, bbox, width=180, height=227):
    """Save cropped and resized images."""
    x, y, w, h = bbox
    img_crop = img[y:h, x:w]
    img_crop = cv2.resize(img_crop, (width, height))
    cv2.imwrite(name + ".jpg", img_crop)

def extract_text_near_substring(input_string, substring):
    # Find the index of the substring
    index = input_string.find(substring)
    if index != -1:
        # Find the start index of the text before the nearest newline
        start_index = input_string.rfind('\n', 0, index)
        if start_index == -1:
            start_index = 0
        else:
            start_index += 1
        
        # Find the end index of the text after the nearest newline
        end_index = input_string.find('\n', index)
        if end_index == -1:
            end_index = len(input_string)
        
        # Extract and return the text
        extracted_text = input_string[start_index:end_index].strip()
        
        # Further processing to include only required parts
        extracted_lines = extracted_text.split('\n')
        # Filter lines containing "BE" and concatenate
        filtered_lines = [line.strip() for line in extracted_lines if substring in line]
        return ' '.join(filtered_lines)
    else:
        return None    

def extract_text(img_path):
    # img = cv2.imread(img_path)
    ocr_text = read_image(img_path) # pytesseract.image_to_string(img)    
    # Define regular expressions to match the patterns
    name_pattern = r"(?<=\n\n)[A-Z\s]+(?=\nBE)"
    name_match = re.search(name_pattern, ocr_text)
    # Extract the matched strings
    if name_match:
        name = name_match.group().strip()
    else:
        name = None
    print("Name:", name)
    substring = "BE"
    output = extract_text_near_substring(ocr_text, substring)
    output=output[7:]
    output=output[:2]+'z'+output[3:]
    print("ID:", output)
    n1=name
    o1=output
    return name,output

def process_image(img_path, new_path):
    """Detect faces in the image, draw rectangles, save cropped faces, and extract text."""
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        print("No faces detected.")
        return

    for counter, face in enumerate(faces):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0, 250, 0), 3)
        save(gray, os.path.join(new_path, str(counter)), (x1, y1, x2, y2))

    print("Done saving faces.")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get('/register', response_class=HTMLResponse)
async def show_register_form(request:Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post('/register', response_class=HTMLResponse)
async def register(request:Request, full_name: str = Form(...), role: str = Form(...), password: str = Form(...)):
    cur = conn.cursor()
    cur.execute("INSERT INTO users (full_name, role, password) VALUES (%s, %s, %s) RETURNING id", (full_name, role, password))
    user_id = cur.fetchone()[0]  # Fetch the new user ID
    conn.commit()
    cur.close()
    return templates.TemplateResponse("message.html", {"request": request, "message":f"Account Created Successfully.<br/>User ID: {user_id}"})

@app.get('/login', response_class=HTMLResponse)
async def show_login_form(request:Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post('/login', response_class=HTMLResponse)
async def login(request:Request, id: str = Form(...), password: str = Form(...)):
    cur = conn.cursor()
    cur.execute("SELECT id, full_name FROM users WHERE full_name = %s AND password = %s", (id, password))
    user = cur.fetchone()
    cur.close()
    if user:
        return RedirectResponse("/dashboard")
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials. Please try again."})

@app.get('/dashboard', response_class=HTMLResponse)
async def show_login_form(request:Request):
    return templates.TemplateResponse("list.html", {"request": request, "register_url": "/dashboard"})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("register_id_card.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename
    file_path = os.path.join('static/uploads', filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    process_image(file_path, 'static/uploads')
    n1,o1 = extract_text(file_path)
    image_path = "static/uploads/0.jpg"  # Change to your image file path
    image = Image.open(image_path)
    binary_data = image.tobytes()
    cursor.execute(
        sql.SQL("INSERT INTO id_info (name, output, image_data) VALUES (%s, %s, %s)"),
        (n1, o1,binary_data)
    )
    conn.commit()
    processed_image_filename  = '0.jpg'
    # Construct the URL path to the processed image
    processed_image_url = '/static/uploads/' + processed_image_filename
    # Pass the processed image URL and the extracted text to the template
    return templates.TemplateResponse("register_id_card.html", {"request": request, "processed_image": processed_image_url, "name": n1, "Rollno": o1})

@app.get("/classes/create", response_class=HTMLResponse)
async def create_class_page():
    return templates.TemplateResponse("create_class.html", {"request": request})

@app.post("/classes/create", response_class=HTMLResponse)
async def create_class(course_id: str = Form(...), student_ids: List[str] = Form(...)):
    student_ids_json = json.dumps(student_ids)
    
    sql_command = sql.SQL("INSERT INTO classes (course_id, student_ids) VALUES (%s, %s)")
    values = (course_id, student_ids_json)
    execute_sql(sql_command, values)
    return templates.TemplateResponse("message.html", {"request": request, "message":f"Class Created Successfully.<br/>Course ID: {course_id}"})

@app.get("/classes/{course_id}", response_class=HTMLResponse)
async def read_class(course_id: str):
    sql_command = sql.SQL("SELECT * FROM classes WHERE course_id = %s")
    values = (course_id,)
    result = execute_sql(sql_command, values, type="SEL")
    print(result)
    if not result:
        raise HTTPException(status_code=404, detail="Class not found")
    return templates.TemplateResponse("view_class.html", {"request": request, "course_id":result[0][0], "students":result[0][2]})

@app.get("/classes/{course_id}/delete", response_class=HTMLResponse)
async def delete_class(course_id: str):
    sql_command = sql.SQL("DELETE FROM classes WHERE course_id = %s")
    values = (course_id,)
    execute_sql(sql_command, values)
    return templates.TemplateResponse("message.html", {"request": request, "message":f"Class deleted successfully."})

@app.get("/classes", response_class=HTMLResponse)
async def view_all_classes():
    sql_command = sql.SQL("SELECT * FROM classes")
    result = execute_sql(sql_command, type="SEL")
    return templates.TemplateResponse("view_classes.html", {"request": request, "result":result})

@app.get("/classes/{course_id}/upload_photo", response_class=HTMLResponse)
async def upload_group_photo_page(course_id: str):
    return templates.TemplateResponse("group.htm", {"request": request, "course_id": course_id})

@app.post("/classes/{course_id}/upload_photo", response_class=HTMLResponse)
async def upload_group_photo(course_id: str, file: UploadFile = File(...)):
    # Save the uploaded photo
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the photo to count headcount and generate UID
    files = {'file': open(file_path, 'rb')}
    response = requests.post("https://localhost:8000/facecount", files=files)
    count = response.json()["face_count"]

    myUid = str((uuid.uuid4())).split("-")[0]

    with conn.cursor() as cursor:
        # Insert data into the database
        cursor.execute(
            """
            INSERT INTO attendance (u_id, roll_number)
            VALUES (%s, %s)
            """,
            (myUid, rollNumber)
        )
        conn.commit()
    
    # Respond with headcount and UID link
    return templates.TemplateResponse("message.html", {"request": request, "message": f"Headcount: {headcount}<br/>Ask people tu submit at /attendance/{myUid}"})

@app.get("/attendance/{uid}", response_class=HTMLResponse)
async def take_selfie(uid: str):
    return templates.TemplateResponse("id.html", {"request": request, "uid": uid})

@app.post("/attendance/{uid}", response_class=HTMLResponse)
async def mark_attendance(uid: str, file: UploadFile = File(...)):
    # Identify the person in the selfie and mark attendance
    files = {'file': open(file_path, 'rb')}
    response = requests.post("https://localhost:8000/recognise", files=files)
    person_identified = response.json()["matched"]

    if person_identified:
        return f"marked attendance for {person_identified}"
    else:
        return "failed to mark attendance"


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()