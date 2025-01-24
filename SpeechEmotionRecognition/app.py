from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Path to the dataset folder (same as used for training)
dataset_folder = r"C:\Users\polot\Downloads\archive (1)\my-Audio-Dataset\Emotions"

# Dynamically generate emotion labels
emotions = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]
emotion_labels = {i: emotion for i, emotion in enumerate(emotions)}
print("Updated Emotion Labels:", emotion_labels)

# Load the trained model
model = tf.keras.models.load_model("speech_emotion_recognition_model.h5")

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html exists in the 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        # Load and preprocess the audio file
        y_audio, sr = librosa.load(file, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_scaled = mfcc_scaled.reshape(1, -1)

        # Predict emotion
        predictions = model.predict(mfcc_scaled)
        predicted_index = np.argmax(predictions)
        print("Predicted Index:", predicted_index)

        if predicted_index not in emotion_labels:
            return jsonify({"error": f"Prediction index {predicted_index} not found in emotion_labels"}), 500

        predicted_emotion = emotion_labels[predicted_index]
        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

