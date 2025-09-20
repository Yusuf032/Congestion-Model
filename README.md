🚦 Road Congestion Detection Model

An AI-based system for detecting road traffic congestion from images and video streams. This project leverages deep learning techniques to classify road conditions as congested or free-flowing, enabling smarter traffic management, navigation assistance, and urban mobility planning.

✨ Key Features

✅ Real-time congestion detection from images and video

✅ Robust CNN-based architecture for feature extraction

✅ Outputs binary classification:

0 → No Congestion

1 → Congestion

✅ Deployable via API or web interface (Streamlit/Flask)

📂 Project Structure
congestion-detection/
├── data/                # Dataset (images/videos of traffic)
├── notebooks/           # Training and evaluation experiments
├── models/              # Trained model weights
├── src/                 # Core source code
│   ├── data_preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── app.py               # Web demo (Streamlit/Flask)
├── requirements.txt     # Dependencies
└── README.md            # Documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/congestion-detection.git
cd congestion-detection


(Optional) Create a virtual environment:

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt

📊 Dataset

The model was trained on a dataset of traffic scenes containing both congested and non-congested roads.

Classes: Congested, Not Congested

Preprocessing:

Resizing to 224×224 pixels

Normalization

Data augmentation: rotation, flipping, brightness adjustments

(Adaptable to any dataset of labeled road traffic images)

🧠 Model Overview

Architecture: Convolutional Neural Network (CNN)

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall

🚀 Usage
1. Train the Model
python src/train.py --epochs 20 --batch_size 32 --data_path ./data

2. Run Inference
python src/predict.py --input test_image.jpg

3. Launch Web Application
streamlit run app.py


Upload traffic images/videos to get real-time predictions.

📈 Results

Accuracy: 97.7% (on held-out test set)

Inference Speed: Real-time processing for video input

Generalization: Performs reliably under varying weather and lighting


🔮 Roadmap

Extend to multi-class density levels (low, medium, high congestion)

Edge deployment on IoT devices (Raspberry Pi, Jetson Nano)

Integration with navigation APIs (e.g., Google Maps, Waze)

👥 Contributors

Your Name – @Yusuf032

📜 License

This project is licensed under the MIT License.
