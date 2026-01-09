import os
import numpy as np
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Paths
BASE_PATH = r"c:\Users\morak\CODE\ANEMIA_PROJ"
EYES_PATH = os.path.join(BASE_PATH, "eyes")
FINGERNAILS_PATH = os.path.join(BASE_PATH, "Fingernails")

IMG_SIZE = 224
SCALER_PATH = os.path.join(BASE_PATH, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_PATH, "classifier_model.pkl")

def extract_features(image_path):
    """Extract features from a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Resize
        img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Color histogram (36 features)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [12], [0, 256])
            features.extend(hist.flatten())
        
        # Color statistics (6 features)
        for i in range(3):
            features.append(np.mean(hsv[:,:,i]))
            features.append(np.std(hsv[:,:,i]))
        
        # Texture features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        features.append(np.mean(edges))
        features.append(np.std(edges))
        features.append(np.sum(edges))
        
        # Brightness (2 features)
        features.append(np.mean(gray))
        features.append(np.std(gray))
        
        return np.array(features)
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def load_data_from_directory(directory_path, label):
    """Load images from a directory (eyes or fingernails)."""
    X = []
    y = []
    
    for class_name in ['Anemic', 'NonAnemic']:
        class_path = os.path.join(directory_path, class_name)
        
        if not os.path.exists(class_path):
            print(f" Directory not found: {class_path}")
            continue
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n Loading {len(images)} images from: {class_name}")
        
        for img_file in tqdm(images):
            img_path = os.path.join(class_path, img_file)
            features = extract_features(img_path)
            
            if features is not None:
                X.append(features)
                # 0 = Non-Anemic, 1 = Anemic
                y.append(0 if class_name == 'NonAnemic' else 1)
    
    return X, y

def main():
    print("\n" + "=" * 70)
    print(" Training Anemia Detection Model")
    print("=" * 70)
    
    # Load data from both eyes and fingernails
    print("\n Loading training data...")
    X_eyes, y_eyes = load_data_from_directory(EYES_PATH, 'eyes')
    X_fingernails, y_fingernails = load_data_from_directory(FINGERNAILS_PATH, 'fingernails')
    
    # Combine datasets
    X = np.array(X_eyes + X_fingernails)
    y = np.array(y_eyes + y_fingernails)
    
    print(f"\n Total images loaded: {len(X)}")
    print(f"   - Anemic: {np.sum(y == 1)}")
    print(f"   - Non-Anemic: {np.sum(y == 0)}")
    
    if len(X) == 0:
        print(" No images found! Check your directory structure.")
        return
    
    # Standardize features
    print("\n Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    print(" Training Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_scaled, y)
    
    # Evaluate on training data
    y_pred = classifier.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n Training Accuracy: {accuracy * 100:.2f}%")
    
    print("\n Classification Report:")
    print(classification_report(y, y_pred, target_names=['Non-Anemic', 'Anemic']))
    
    # Save model
    print("\n Saving model...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f" Model saved to: {MODEL_PATH}")
    print(f" Scaler saved to: {SCALER_PATH}")
    
    # Plot confusion matrix
    print("\n Generating confusion matrix...")
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Anemic', 'Anemic'],
                yticklabels=['Non-Anemic', 'Anemic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(BASE_PATH, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    print(" Confusion matrix saved!")
    
    print("\n" + "=" * 70)
    print(" Training complete! Ready to use the web app.")
    print("=" * 70)

if __name__ == '__main__':
    main()
