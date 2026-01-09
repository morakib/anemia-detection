# 🏥 Anemia Detection - ML Web Application

A machine learning web application that detects anemia from eye and fingernail images using scikit-learn and Flask.

## 📋 Features

- **Binary Classification**: Anemic vs Non-Anemic detection
- **Dual Image Support**: Works with both eye and fingernail images
- **Smart Feature Extraction**: Color histograms, texture analysis, brightness detection
- **Web Interface**: Simple, responsive UI for image upload and predictions
- **High Accuracy**: Random Forest classifier trained on combined dataset

## 🎯 Model Details

### Training Approach
- **Algorithm**: Random Forest (100 trees)
- **Features**: 51 hand-crafted features per image
- **Data**: Combined eyes and fingernails images
- **Accuracy**: High generalization on new images

### Features Extracted
- Color distribution (HSV histograms)
- Color statistics (mean, std deviation)
- Texture analysis (edge detection)
- Brightness and contrast metrics

## 📂 Project Structure

```
anemia_model/
├── app.py                      # Flask web application
├── train_sklearn_model.py      # Model training script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── templates/
│   └── index.html             # Web interface
│
├── eyes/
│   ├── Anemic/                # Eye images of anemic patients
│   └── NonAnemic/             # Eye images of non-anemic people
│
└── Fingernails/
    ├── Anemic/                # Fingernail images of anemic patients
    └── NonAnemic/             # Fingernail images of non-anemic people
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/anemia-detection.git
cd anemia-detection
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_sklearn_model.py
```
This will:
- Load all images from `eyes/` and `Fingernails/` directories
- Extract features from each image
- Train the Random Forest classifier
- Save `classifier_model.pkl` and `scaler.pkl`

### 5. Run the Web App
```bash
python app.py
```

Open your browser to: **http://localhost:5000**

## 📤 How to Use

1. **Open the web app** in your browser
2. **Upload an image** - eye or fingernail (JPG, PNG, GIF, BMP)
3. **Get instant prediction**:
   - Anemic / Non-Anemic
   - Confidence percentage
   - Probability scores

## 📊 Training Data

You need to organize images in this structure:

```
eyes/
├── Anemic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── NonAnemic/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

Fingernails/
├── Anemic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── NonAnemic/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Minimum training data**: 50-100 images per class (total ~200-400 images)

## 🔧 Requirements

See `requirements.txt` for complete list:
- Flask (web framework)
- scikit-learn (ML model)
- OpenCV (image processing)
- Pillow (image loading)
- NumPy, Pandas (data processing)

## 📈 Model Performance

After training, the script outputs:
- Overall accuracy
- Per-class precision/recall/F1-score
- Confusion matrix visualization

Example:
```
Anemic:     Precision: 0.88  Recall: 0.92  F1-Score: 0.90
Non-Anemic: Precision: 0.91  Recall: 0.87  F1-Score: 0.89
```

## 🌐 Deployment Options

### Local Deployment
- Run `python app.py` on your machine
- Access via `http://localhost:5000`

### Cloud Deployment

#### Option 1: Heroku (Free tier available)
```bash
pip install gunicorn
echo "web: gunicorn app:app" > Procfile
echo "python-3.9.6" > runtime.txt
git push heroku main
```

#### Option 2: PythonAnywhere
1. Sign up at pythonanywhere.com
2. Upload repository
3. Configure web app settings
4. Done!

#### Option 3: Railway/Render
- Connect GitHub repo
- Automatic deployment on push
- Zero configuration needed

## ⚠️ Limitations

- Model trained on specific image types - accuracy depends on similar images
- Requires good image quality
- Best results with full eye or fingernail images
- Single image prediction (batch processing available on request)

## 🔐 Medical Disclaimer

⚠️ **THIS IS NOT A MEDICAL DEVICE**

This tool is for educational/research purposes only. It should NOT be used for:
- Clinical diagnosis
- Medical treatment decisions
- Replacing professional medical advice

Always consult qualified healthcare professionals for medical diagnosis.

## 📝 License

[Choose your license - MIT, Apache 2.0, etc.]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit a pull request

## 📧 Contact

For questions or issues, open a GitHub issue or contact [your-email].

## 📚 References

- Anemia indicators in eyes and fingernails
- Random Forest classifier documentation
- Flask web framework guide

---

**Made with ❤️ for medical ML education**
