import cv2 #openCV
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
from io import BytesIO

# Load the trained model
model = load_model('fashion_classifier_model.h5')

# Preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Capture live video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read each frame from the video stream

    # Display the captured frame
    cv2.imshow('Webcam', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Process the captured frame
    input_image = preprocess_image(frame)
    prediction = model.predict(input_image)
    class_label = "saree" if prediction[0][0] < 0.5 else "kediya"
    print("Predicted class: ", class_label)

# Release the video capture
cap.release()

# Close the OpenCV window (add a delay to ensure it's properly closed)
cv2.waitKey(1000)
cv2.destroyAllWindows()
