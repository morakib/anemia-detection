import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_PATH = r"c:\Users\morak\CODE\ANEMIA_PROJ"
EYES_PATH = os.path.join(BASE_PATH, "eyes")
FINGERNAILS_PATH = os.path.join(BASE_PATH, "Fingernails")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "anemia_model.pth")
METRICS_SAVE_PATH = os.path.join(BASE_PATH, "training_metrics.txt")

class AnemiaDataset(Dataset):
    """Custom Dataset for loading anemia images."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None

def load_images_from_directory(directory):
    """Load all images from a directory and subdirectories."""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(['Anemic', 'NonAnemic']):
        class_path = os.path.join(directory, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory not found: {class_path}")
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(class_path, img_file)
                    image_paths.append(img_path)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    return image_paths, np.array(labels)

def split_data(image_paths, labels):
    """Split data into train and validation sets."""
    indices = np.random.permutation(len(image_paths))
    image_paths = np.array(image_paths)[indices]
    labels = labels[indices]
    
    split_idx = int(len(image_paths) * (1 - VALIDATION_SPLIT))
    
    train_paths = image_paths[:split_idx].tolist()
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:].tolist()
    val_labels = labels[split_idx:]
    
    return (train_paths, train_labels), (val_paths, val_labels)

def build_model():
    """Build transfer learning model using ResNet50."""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    
    # Replace final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    return model

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), correct / total, all_preds, all_labels

def main():
    print("=" * 60)
    print("ANEMIA DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load images from eyes folder
    print("\n[1/4] Loading eye images...")
    eyes_paths, eyes_labels = load_images_from_directory(EYES_PATH)
    print(f"  - Loaded {len(eyes_paths)} eye images")
    print(f"    Anemic: {sum(eyes_labels == 0)}, NonAnemic: {sum(eyes_labels == 1)}")
    
    # Load images from fingernails folder
    print("\n[2/4] Loading fingernail images...")
    nails_paths, nails_labels = load_images_from_directory(FINGERNAILS_PATH)
    print(f"  - Loaded {len(nails_paths)} fingernail images")
    print(f"    Anemic: {sum(nails_labels == 0)}, NonAnemic: {sum(nails_labels == 1)}")
    
    # Combine datasets
    print("\n[3/4] Combining datasets...")
    all_paths = eyes_paths + nails_paths
    all_labels = np.concatenate([eyes_labels, nails_labels])
    print(f"  - Total images: {len(all_paths)}")
    print(f"    Total Anemic: {sum(all_labels == 0)}")
    print(f"    Total NonAnemic: {sum(all_labels == 1)}")
    
    # Prepare training and validation data
    print("\n[4/4] Preparing data...")
    (train_paths, train_labels), (val_paths, val_labels) = split_data(all_paths, all_labels)
    print(f"  - Training set: {len(train_paths)} images")
    print(f"  - Validation set: {len(val_paths)} images")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AnemiaDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AnemiaDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model
    print("\n[5/5] Building model...")
    model = build_model().to(DEVICE)
    print(f"Using device: {DEVICE}")
    
    # Loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nTraining model...")
    print("-" * 60)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_true = validate(model, val_loader, loss_fn, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  * Best model saved!")
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    _, train_acc, train_preds, train_true = validate(model, train_loader, loss_fn, DEVICE)
    _, val_acc, val_preds, val_true = validate(model, val_loader, loss_fn, DEVICE)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print("\n" + "-" * 60)
    print("Validation Set Classification Report:")
    print("-" * 60)
    print(classification_report(
        val_true, val_preds,
        target_names=['Anemic', 'NonAnemic']
    ))
    
    # Save metrics to file
    with open(METRICS_SAVE_PATH, 'w') as f:
        f.write("ANEMIA DETECTION MODEL - TRAINING METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Images: {len(all_paths)}\n")
        f.write(f"  - Eye Images: {len(eyes_paths)}\n")
        f.write(f"  - Fingernail Images: {len(nails_paths)}\n\n")
        f.write(f"Training Set Size: {len(train_paths)}\n")
        f.write(f"Validation Set Size: {len(val_paths)}\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  - Input Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Learning Rate: {LEARNING_RATE}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  - Training Accuracy: {train_acc:.4f}\n")
        f.write(f"  - Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"  - Best Validation Accuracy: {best_val_acc:.4f}\n")
    
    print(f"\nMetrics saved to: {METRICS_SAVE_PATH}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {os.path.join(BASE_PATH, 'training_history.png')}")
    
    # Confusion Matrix
    cm = confusion_matrix(val_true, np.array(val_preds).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Anemic', 'NonAnemic'],
                yticklabels=['Anemic', 'NonAnemic'])
    plt.title('Validation Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {os.path.join(BASE_PATH, 'confusion_matrix.png')}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
