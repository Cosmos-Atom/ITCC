import pandas as pd #for analysis
import numpy as np  #numerical operations
from tensorflow.keras.preprocessing import image #imageprocession
from tensorflow.keras.applications.vgg16 import preprocess_input # preprocessor input
from tensorflow.keras.models import Sequential #sequential model
from tensorflow.keras.layers import Dense, Flatten #connected layers
from sklearn.model_selection import train_test_split # split the data into training and testing
from PIL import Image #image processing
import requests #requesting the urls
from io import BytesIO #images from urls
from sklearn.preprocessing import LabelEncoder #labels
from tensorflow.keras.utils import to_categorical # hotencoded vectors


#Reading from the CSV file
data = pd.read_csv('data.csv')
image_urls = data['Image'].values
labels = data['Type'].values

# Encode labels as numeric values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoded vectors
labels = to_categorical(labels)

def preprocess_images(image_urls):
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        images.append(img)
    return np.array(images)

processed_images = preprocess_images(image_urls)

X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

model.save('fashion_classifier_model.h5')

