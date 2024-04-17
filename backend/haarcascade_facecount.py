import cv2
import sys
from Face_Recog import facerecognition


def main(imagePath="null"):
    # Get user supplied values
    if imagePath == "null":
        imagePath = sys.argv[1]
    # imagePath = "10007.jpg"  # Use this line for testing with a hardcoded image path

    # Load the cascade classifier for face detection
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=4,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw rectangles around the faces in the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Resize the image for display
    resized_image = cv2.resize(image, (1024, 540))  # Adjust the size as needed

    # Display the original image with rectangles around the faces
    # cv2.imshow("Faces found", resized_image)
    # Wait for 10 seconds (10000 milliseconds)
    cv2.waitKey(5000)
    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Write the image with rectangles to the file system
    status = cv2.imwrite('saved.jpg', image)
    print("Image written to file-system:", status)

    # facerecognition.main()

    cv2.waitKey(0)

    return [format(len(faces)), "./saved.jpg"]


if __name__ == "__main__":
    main()