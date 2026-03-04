# Sign Language Translator

A real-time sign language recognition system using Deep Learning (LSTM) and Computer Vision (MediaPipe). This project translates hand gestures into text sentences with automated grammar correction.

## 🚀 Key Features

- **Real-time hand tracking** using MediaPipe Holistic pipeline.
- **Deep Learning Model**: LSTM-based architecture for sequence prediction.
- **Custom Data Collection**: Build your own dataset of signs easily.
- **Grammar Correction**: Integrated NLP support via `language-tool-python`.
- **User-Friendly**: Simple scripts for data collection, training, and testing.

## 📂 Project Structure

- `data_collection.py`: Script to record hand gestures and save landmarks.
- `model.py` / `lstm.py`: Model architecture and training logic.
- `main.py`: The core application for real-time translation.
- `my_functions.py`: Utility functions for landmark extraction and visualization.

## 🛠️ Prerequisites

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow / Keras
- `language-tool-python` (Requires Java 8+)

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Sign-Language-Translator
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe tensorflow language-tool-python
   ```

3. (Optional) Run the grammar tool setup:
   ```python
   # This happens automatically on the first run of language-tool-python
   ```

## 📖 Usage

### 1. Data Collection
Run `data_collection.py` to record gestures for specific signs. Customize the `actions` list in the script to add new words.

### 2. Training
Run the training script (e.g., `lstm.py` or `model.py`) to train the LSTM model on your collected data.

### 3. Real-time Prediction
Run `main.py` to start the translator.
- **Enter**: Perform grammar check on the current sentence.
- **Space**: Reset the sentence.
- **Q**: Quit the application.

---

## 📄 License
This project is licensed under the MIT License.
This project is based on the original work by dgovor:
https://github.com/dgovor/Sign-Language-Translator

Modifications and improvements by IppiliSahasra08.
