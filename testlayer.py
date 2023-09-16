import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vit_pytorch import ViT
import optuna
import h5py
import cv2
import numpy as np
from PIL import Image

torch.manual_seed(1000)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Load the Galaxy10 dataset from an h5 file
def load_galaxy10_data(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        labels = f['ans'][:]
    return images, labels

# Define a custom dataset class
class Galaxy10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the dataset and split it into train, validation, and test sets
images, labels = load_galaxy10_data('../Galaxy10.h5')

# Define data augmentation and transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Replace with actual mean and std
])
X_train, X_tmp, y_train, y_tmp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

train_dataset = Galaxy10Dataset(X_train, y_train, transform=transform)
val_dataset = Galaxy10Dataset(X_val, y_val, transform=transform)
test_dataset = Galaxy10Dataset(X_test, y_test, transform=transform)

# Define an Optuna objective function to optimize hyperparameters
def objective(trial):
    print(f"Trial {trial.number}: Hyperparameter tuning in progress...")
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])  # Adjust batch size options as needed

    # Create data loaders with the chosen batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)

    # Initialize the ViT model with self-attention
    model = ViT(
        image_size=(224,224),
        patch_size=16,
        num_classes=10,
        dim=512,
        depth=12,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    num_epochs = 1
    best_accuracy = 0.0

    # Log file for training output
    log_file = open('training_log.txt', 'w')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_outputs = model(inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_predictions.extend(val_preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_predictions)
        log_message = f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}'
        print(log_message)
        log_file.write(log_message + '\n')

        # Save the model if it's the best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    log_file.close()
    return -best_accuracy  # Optuna minimizes the objective, so we negate accuracy

# Optuna hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)  # You can adjust the number of trials as needed

# Get the best hyperparameters
best_params = study.best_params
best_learning_rate = best_params['learning_rate']
best_weight_decay = best_params['weight_decay']
best_batch_size = best_params['batch_size']

print(f'Best Learning Rate: {best_learning_rate:.6f}')
print(f'Best Weight Decay: {best_weight_decay:.6f}')
print(f'Best Batch Size: {best_batch_size}')

# Train the model with the best hyperparameters
best_model = ViT(
    image_size=(224,224),
    patch_size=16,
    num_classes=10,
    dim=512,
    depth=12,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1,
).to(device)

best_optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)

# Create data loaders with the best batch size
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size,num_workers=4)

num_epochs = 1
best_accuracy = 0.0

# Log file for training output
log_file = open('training_log.txt', 'a')

for epoch in range(num_epochs):
    print('in the best model training loop right now')
    best_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        best_optimizer.zero_grad()
        outputs = best_model(inputs)
        criterion=nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        best_optimizer.step()
        running_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_loss = running_loss / len(train_loader)

    # Validation
    best_model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            val_outputs = best_model(inputs)
            _, val_preds = torch.max(val_outputs, 1)
            val_predictions.extend(val_preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_targets, val_predictions)
    log_message = f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}'
    print(log_message)
    log_file.write(log_message + '\n')

    # Save the model if it's the best so far
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(best_model.state_dict(), 'best_model.pth')

log_file.close()

# Load the best model
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()

#summary(best_model, (3, 224, 224))
print(type(best_model.transformer))
print('reached here')

# Initialize a list to store attention maps
attention_maps = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass to get the logits
        test_outputs = best_model(inputs)

        # Get the attention maps from the self-attention layers in the transformer
        attention = []
        for layer in best_model.transformer.layers:
            # Assuming your Attention module has an attribute 'attention_map'
            attention_map = layer[0].Linear
            attention.append(attention_map)

        attention = torch.stack(attention)  # Stack attention maps from all layers

        attention_maps.extend(attention.cpu().numpy())

# Continue with the rest of your code to overlay attention maps on input images
# ...

# Save attention maps overlaid on input images
for i in range(len(test_loader.dataset)):
    image = X_test[i]
    attention_map = attention_maps[i]

    # Normalize attention map values to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Resize the attention map to match the image size
    attention_map = np.uint8(255 * attention_map)
    attention_map = np.array(Image.fromarray(attention_map).resize((image.shape[1], image.shape[0])))

    # Create a heatmap by overlaying the attention map on the image
    heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_image = cv2.addWeighted(image.transpose(1, 2, 0), 0.7, heatmap, 0.3, 0)

    # Save the overlayed image
    image_filename = f'attention_overlay_{i}.png'
    cv2.imwrite(image_filename, cv2.cvtColor((overlayed_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Classification report
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
classification_rep = classification_report(test_targets, test_predictions, target_names=class_names)

# Save outputs to a file
with open('output.txt', 'w') as output_file:
    output_file.write("Test Accuracy: {:.4f}\n".format(test_accuracy))
    output_file.write("\nClassification Report:\n")
    output_file.write(classification_rep)

print("Outputs saved to 'output.txt' and attention maps saved as images.")
print("Reached the end")

