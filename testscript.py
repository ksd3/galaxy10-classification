import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import optuna
import h5py
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(10)

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
images, labels = load_galaxy10_data('Galaxy10.h5')

# Define data augmentation and transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Initialize the model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 10 output classes for Galaxy10
    model.to(device)

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
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100):
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
best_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = best_model.fc.in_features
best_model.fc = nn.Linear(num_ftrs, 10)
best_model.to(device)
best_optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)

# Create data loaders with the best batch size
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, num_workers=4)

num_epochs = 2
best_accuracy = 0.0

# Log file for training output
log_file = open('training_log.txt', 'a')

for epoch in range(num_epochs):
    print('In the best model training loop right now')
    best_model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100):
        inputs, labels = inputs.to(device), labels.to(device)
        best_optimizer.zero_grad()
        outputs = best_model(inputs)
        criterion = nn.CrossEntropyLoss()
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

# Testing
test_predictions = []
test_targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        test_outputs = best_model(inputs)
        _, test_preds = torch.max(test_outputs, 1)
        test_predictions.extend(test_preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_targets, test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Classification report
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
classification_rep = classification_report(test_targets, test_predictions, target_names=class_names)

# Save outputs to files
with open('output.txt', 'w') as output_file:
    output_file.write("Test Accuracy: {:.4f}\n".format(test_accuracy))
    output_file.write("\nClassification Report:\n")
    output_file.write(classification_rep)

# Plot weights norm and save to a file
def plot_weights_norm(model):
    weight_norms = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_norm = param.norm().item()
            weight_norms.append(weight_norm)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(weight_norms)), weight_norms, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Weight Norm')
    plt.title('Weight Norms for Each Layer')
    plt.grid(True)
    plt.savefig('weight_norms.png')  # Save the plot as an image

# Call the function to plot weight norms
plot_weights_norm(best_model)

# Choose an image from the test dataset (replace 0 with the index of the image you want)
fixed_image, _ = test_dataset[0]
fixed_image = fixed_image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU if available

# Get feature maps from the model
def get_intermediate_feature_maps(model, image):
    activation = {}

    def hook(name):
        def hook_fn(module, input, output):
            activation[name] = output
        return hook_fn

    hooks = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            hook_fn = hook(name)
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(image)

    for hook in hooks:
        hook.remove()

    return activation

feature_maps = get_intermediate_feature_maps(best_model, fixed_image)

# Visualize the feature maps (you can choose a specific layer)
layer_name = 'conv1'  # Replace with the layer name you want to visualize
feature_map = feature_maps[layer_name].cpu().squeeze(0)  # Remove batch dimension and move to CPU

# Visualize the feature maps in a grid
def visualize_feature_maps_grid(feature_map):
    num_feature_maps = feature_map.size(0)
    rows = int(np.sqrt(num_feature_maps))
    cols = int(np.ceil(num_feature_maps / rows))
    
    plt.figure(figsize=(12, 12))
    for i in range(num_feature_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[i], cmap='viridis')  # You can choose a different colormap
        plt.title(f'Feature Map {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps_grid.png')  # Save the grid of feature maps as an image

# Call the function to visualize feature maps in a grid
visualize_feature_maps_grid(feature_map)

print("Outputs saved to 'output.txt', 'weight_norms.png', 'feature_maps.png', and 'feature_maps_grid.png'")
print("Reached the end")

