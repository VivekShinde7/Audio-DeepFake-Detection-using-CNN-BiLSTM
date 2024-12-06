# Audio DeepFake Detection using CNN-BiLSTM

## APP Demo

https://github.com/user-attachments/assets/2e036a20-97fc-4b13-accb-63e7430f7e57

## Overview
This project aims to detect audio deepfakes using a hybrid approach that combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory Networks (BiLSTM). The system is designed to effectively classify audio data into genuine or fake categories, offering a robust solution to the growing challenges posed by audio-based misinformation.

---

## Key Features
- **Hybrid Model Architecture**: Combines the feature extraction power of CNNs with the sequential processing capabilities of BiLSTMs.
- **State-of-the-Art Accuracy**: Achieves high detection accuracy, making it suitable for practical applications.
- **Research Contribution**: Includes detailed insights and a research paper explaining the methodology and findings.

---

## Table of Contents
- [Project Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Dataset 
- For Real Audio : https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset
- For Fake Audio : https://www.kaggle.com/datasets/andreadiubaldo/wavefake-test
  - The dataset includes audio recordings, labeled as either genuine or deepfake.
  - Preprocessing steps involve:
    - Feature extraction using Mel-frequency cepstral coefficients (MFCCs).
    - Data augmentation techniques to enhance model robustness.


---

## Model Architecture
The model leverages the strengths of:
1. **CNN**:
   - Extracts spatial features from MFCCs.
   - Efficiently identifies patterns and anomalies.
2. **BiLSTM**:
   - Processes sequential data to capture temporal dependencies.
   - Bidirectional design ensures both past and future context is utilized.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/VivekShinde7/Audio-DeepFake-Detection-using-CNN-BiLSTM.git
   cd Audio-DeepFake-Detection-using-CNN-BiLSTM
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run app.py:
   ```bash
   streamlit run app.py
   ```

---

## Results
- **Performance Metrics**:
  - Accuracy: **98.3%**
  - Precision: **97.8%**
  - Recall: **98.8%**
- Visualization of confusion matrix, System Architecture & Evaluation is available in the `results` folder.

---

## Future Work
- Enhance the dataset to include diverse languages and accents.
- Optimize the model for real-time detection.
- Explore the integration of transformer-based architectures like Wav2Vec2.0.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

---

## Acknowledgments
- Special thanks to open-source contributors and dataset providers.
- Inspiration drawn from advancements in audio deepfake detection research.

For queries or suggestions, feel free to open an issue or contact [Vivek Shinde](https://github.com/VivekShinde7).
