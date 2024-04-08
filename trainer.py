import torch
from util import getImagesAndLables
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import Adam, lr_scheduler
import ssl
import random

ssl._create_default_https_context = ssl._create_unverified_context

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.label_mapping = {"do": 0, "re": 1, "mi": 2, "fa": 3, "sol": 4, "la": 5, "ti": 6}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label = self.label_mapping[label]
        label = torch.tensor(label)
            
        return image, label

transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

word_list = ["do", "re", "mi", "fa", "sol", "la", "ti"]
wordToLabel = {word: index for index, word in enumerate(word_list)}
labelToWord = {index: word for index, word in enumerate(word_list)}

images, labels = getImagesAndLables()
myDataset = CustomDataset(images, labels, transform=transform)
train_size = int(0.8 * len(myDataset))
val_size = len(myDataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(myDataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

resnet34 = models.resnet34(pretrained=True)
num_ftrs = resnet34.fc.in_features
resnet34.fc = torch.nn.Linear(num_ftrs, len(word_list))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet34.to(device)

optimizer = Adam(resnet34.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
best_val_loss = float('inf')
best_model_path = "best_resnet34_model.pth"

for epoch in range(epochs):
    resnet34.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = resnet34(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    resnet34.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet34(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")

    if val_loss < best_val_loss:
        torch.save(resnet34.state_dict(), best_model_path)
        best_val_loss = val_loss

    scheduler.step()

resnet34.load_state_dict(torch.load(best_model_path))

for _ in range(10):
    random_index = random.randint(0, len(val_dataset) - 1)
    test_image, test_label = val_dataset[random_index]
    test_image_tensor = test_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet34(test_image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()
    predicted_word = labelToWord[predicted_label]
    print("Actual label:", labelToWord[test_label.item()])
    print("Predicted label:", predicted_word)
