from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

def decode_base64_image(base64_string):
    if base64_string:
        try:
            # Remove the "data:image/*;base64," prefix if it exists
            if base64_string.startswith('data:image'):
                _, base64_string = base64_string.split(',', 1)

            decoded = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(decoded))
            return img

        except Exception as e:
            return None
        
def detect_faces_and_get_image(base64_string):
    # Initialize the MTCNN detector
    mtcnn = MTCNN()

    try:
        if base64_string.startswith('data:image'):
            _, base64_string = base64_string.split(',', 1)

        decoded = base64.b64decode(base64_string)

        # Load the image using PIL
        img = Image.open(io.BytesIO(decoded))

        # Convert PIL image to NumPy array
        image_np = np.array(img)

        # Detect faces and facial landmarks using MTCNN
        faces = mtcnn.detect_faces(image_np)

        if len(faces) == 0:
            print("No face detected. Try again")
            return False
        else:
            return True
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def preprocess_image(image, res):
    #preprocessing function for model
    resized_image = tf.image.resize(image, (res, res))
    preprocessed_image = resized_image / 255.0  # Scale between 0 and 1
    return preprocessed_image


model = load_model('AcneSeverity_MobileNetV2_V2_100epoch.h5')
model2 = load_model('skin_type_v2_classifier.h5')

@app.route('/')
def home():
    return "Home"

@app.route('/severity', methods=['POST'])
def getSeverity():
    try:
        data = request.get_json()
        rside_base64 = data.get('rSide', '')
        lside_base64 = data.get('lSide', '')
        front_base64 = data.get('front', '')

        # Convert base64 to images (PIL.Image)
        rside_image = decode_base64_image(rside_base64)
        lside_image = decode_base64_image(lside_base64)
        front_image = decode_base64_image(front_base64)

        
        face = detect_faces_and_get_image(front_base64)

        if(not face):
            return jsonify({"error": "Face not recognized"}), 500

        #processing with the images here
        preprocessed_rside = preprocess_image(rside_image, 256)
        preprocessed_lside = preprocess_image(lside_image,256)
        preprocessed_front = preprocess_image(front_image, 224)

        resultRSide = model.predict(np.expand_dims(preprocessed_rside, 0))
        resultLSide = model.predict(np.expand_dims(preprocessed_lside, 0))
        resultFront = model2.predict(np.expand_dims(preprocessed_front, 0))

        print(resultRSide)
        print(resultLSide)
        print(resultFront)

        # Find the index of the highest probability in each result
        index1 = np.argmax(resultRSide)
        index2 = np.argmax(resultLSide)
        index3 = np.argmax(resultFront)

        skinType = ''

        if(index3 == 0):
            skinType = 'Dry'
        elif (index3 == 1):
            skinType = 'Normal'
        else:
            skinType = 'Oily'

        print(index1)
        print(index2)

        if(index1 > index2):
            # Respond with a success message or result
            return jsonify({"message": "Images received and processed successfully", "severity": int(index1) + 1, "type" : skinType})
        else:
            # Respond with a success message or result
            return jsonify({"message": "Images received and processed successfully", "severity": int(index2) + 1, "type" : skinType})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


