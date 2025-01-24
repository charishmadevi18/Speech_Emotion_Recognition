Speech Emotion Recognition

This project is a Speech Emotion Recognition system that recognizes emotions (happy, sad, angry, etc.) from Telugu audio files using a machine learning model. The project processes audio files, extracts features, trains a model, and saves the trained model for future use.

Project Structure

app.py: Main application script containing code for loading and processing audio files, extracting features, and training the model.

speech_emotion_recognition_model.h5: Pre-trained model file used for emotion recognition.

templates/: Contains HTML and CSS files for the frontend interface (if applicable).

Features

Emotion Detection: Recognizes emotions from Telugu audio files.

Feature Extraction: Utilizes Mel-frequency cepstral coefficients (MFCC) for feature extraction.

Model Training: Trains a neural network model on extracted features to classify emotions.

Model Saving: Saves the trained model for future use.

How It Works

Dataset Preparation: The dataset consists of subfolders, each representing a different emotion, containing .wav audio files.

Feature Extraction: The script uses librosa to load audio files and extract MFCC features.

Model Training: A neural network is trained on the extracted features using TensorFlow/Keras.

Emotion Classification: The trained model 
