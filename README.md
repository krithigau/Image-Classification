# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

![image](https://github.com/user-attachments/assets/dec2e8ae-8232-482b-b030-a5185cf51738)


## Neural Network Model

![image](https://github.com/user-attachments/assets/2254dcdc-73bc-4bd4-8567-fe7bdc9e91bd)

## DESIGN STEPS

STEP 1: Problem Statement

Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection

Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture

Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization

Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: KRITHIGA U
### Register Number:212223240076
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # write your code here
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

```
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```

# Train the Model

def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print('Name:KRITHIGA U')
        print('Register Number: 212223240076')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

![Screenshot 2025-04-05 093446](https://github.com/user-attachments/assets/0f0d1984-f63d-4e42-a5ce-d9d3e503c078)


### Confusion Matrix

![Screenshot 2025-04-05 093532](https://github.com/user-attachments/assets/ba7a8201-b8c9-4e1e-bf0e-b5b5b82c356b)

### Classification Report

![Screenshot 2025-04-05 093543](https://github.com/user-attachments/assets/154d4f62-513e-4a75-acf8-a63991c43a10)


### New Sample Data Prediction

![Screenshot 2025-04-05 093605](https://github.com/user-attachments/assets/78f838e6-94ef-48fe-b45b-63c64b73dd77)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
