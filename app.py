import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io
import os

import tensorflow as tf

# Function to extract MFCC features for uploaded audio
def extract_features_from_audio(audio_bytes, max_length=500, sr=16000, n_mfcc=40):
    try:
        # Convert bytes to audio array
        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes), 
            sr=sr
        )
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or trim the feature array to a fixed length
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        
        # Reshape to match model input
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def main():
    st.title('Audio Deepfake Detection')
    st.write('Upload an audio file to detect if it is real or deepfake')

    # Load the pre-trained model
    @st.cache_resource
    def load_model():
        try:
            model = tf.keras.models.load_model(r'F:\Audio_Deepfake_Project\savedmodels\updated_model.keras')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload an audio file to check if it's a deepfake"
    )

    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Process and predict
        if st.button('Detect Deepfake'):
            if model is not None:
                # Extract features
                features = extract_features_from_audio(uploaded_file.getvalue())
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict(features)
                    confidence = prediction[0][0]
                    
                    # Determine result
                    is_deepfake = confidence > 0.5
                    
                    # Display results
                    st.subheader('Detection Results')
                    if is_deepfake:
                        st.error(f'🚨 Deepfake Detected (Confidence: {confidence*100:.2f}%)')
                    else:
                        st.success(f'✅ Real Audio (Confidence: {(1-confidence)*100:.2f}%)')
                    
                    # Visualization of confidence
                    st.progress(float(confidence))
            else:
                st.error("Model could not be loaded. Please check the model file.")

    # Sidebar with additional information
    st.sidebar.header('About')
    st.sidebar.info(
        "This app uses a deep learning model to detect audio deepfakes. "
        "It analyzes the audio file's MFCC features to classify if the audio is real or synthetic."
    )

    st.sidebar.header('How it Works')
    st.sidebar.write(
        "1. Upload an audio file\n"
        "2. Click 'Detect Deepfake'\n"
        "3. Get instant results with confidence percentage"
    )

if __name__ == '__main__':
    main()