# 1. Defining the Neural Network (`nn.Module`)

The core structure always inherits from `nn.Module`. You must define two methods: `__init__` (layers) and `forward` (connections).

### **A. Common Layers & Arguments**

| Layer | Syntax & Key Arguments | Usage |
| :--- | :--- | :--- |
| **Convolution** | `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` | Extracting features from images. |
| **Fully Connected** | `nn.Linear(in_features, out_features)` | Classifying features at the end of the network. |
| **Pooling** | `nn.MaxPool2d(kernel_size, stride=None)` | Reducing image size (downsampling). |
| **Dropout** | `nn.Dropout(p=0.5)` | Randomly zeroing neurons to prevent overfitting. |
| **Flatten** | `nn.Flatten()` | Converts 2D/3D maps into 1D vector (for Linear layer). |

### **B. Activation Functions (Alternatives)**

| Function | Syntax | When to use |
| :--- | :--- | :--- |
| **ReLU** | `nn.ReLU()` | Standard for hidden layers. $f(x) = \max(0, x)$. |
| **Sigmoid** | `nn.Sigmoid()` | **Binary Classification** output (0 to 1). |
| **Softmax** | `nn.Softmax(dim=1)` | **Multi-class Classification** output (probabilities sum to 1). |
| **Tanh** | `nn.Tanh()` | Hidden layers (rare now, outputs -1 to 1). |

### **C. Example Code Structure**

```python
import torch
import torch.nn as nn

class ExamModel(nn.Module):
    def __init__(self):
        super(ExamModel, self).__init__()
        
        # 1. Feature Extraction (Conv Layers)
        # Input: (Batch, 1, 28, 28) -> Output: (Batch, 8, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        
        # 2. Activation & Pooling
        self.relu = nn.ReLU()
        # Input: (Batch, 8, 28, 28) -> Output: (Batch, 8, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Classifier (Linear Layers)
        # Flatten size = channels * height * width = 8 * 14 * 14
        self.fc = nn.Linear(in_features=8*14*14, out_features=1)
        
        # 4. Final Activation (Alternative: Sigmoid for Binary)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        # FLATTEN TRICK: .view(batch_size, -1)
        out = out.view(out.size(0), -1) 
        
        out = self.fc(out)
        out = self.sigmoid(out) # Output is between 0 and 1
        return out
```

### 2. Loss Functions (`criterion`)
The loss function measures how "wrong" the model is.

| Task | Loss Function | Key Arguments | Note |
| :--- | :--- | :--- | :--- |
| **Multi-Class** | `nn.CrossEntropyLoss()` | `weight=None`, `reduction='mean'` | **Do NOT** use Softmax in `forward()`; this function does it internally. |
| **Binary Class** | `nn.BCELoss()` | `reduction='mean'` | **Must** use Sigmoid in `forward()`. Targets must be float. |
| **Binary (Stable)**| `nn.BCEWithLogitsLoss()`| `pos_weight=None` | Combines Sigmoid + BCELoss. **Do NOT** use Sigmoid in `forward()`. |
| **Regression** | `nn.MSELoss()` | `reduction='mean'` | Mean Squared Error. For predicting numbers (e.g., price). |  

### 3. Optimizers (`optim`)

The optimizer updates the weights (`parameters`) based on gradients.

### **A. SGD (Stochastic Gradient Descent)**
Classic, simple, often used with momentum.
```python
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01,           # Learning Rate (step size)
    momentum=0.9,      # Accelerates SGD in the relevant direction
    weight_decay=1e-4  # L2 Regularization (penalty for large weights)
)
```

### **B. Adam (Adaptive Moment Estimation)**
Most popular general-purpose optimizer. Adjusts learning rate automatically.
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,          # Usually lower than SGD (e.g., 1e-3 or 3e-4)
    betas=(0.9, 0.999),# Coefficients used for computing running averages
    eps=1e-8,          # Term added to denominator to improve numerical stability
    weight_decay=0     # L2 Regularization
)
```
### 4. The Training Loop (The "Recipe")

You must memorize this 5-step sequence inside the loop.

### **Definitions**
* **Epoch:** One pass through the *entire* dataset.
* **Batch:** A subset of data processed at once.
* **Iteration:** One update step (Forward + Backward).

### **Complete Code Skeleton**

```python
# 1. Setup
model = ExamModel()
criterion = nn.BCELoss() # Binary classification example
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. Loops
num_epochs = 10

for epoch in range(num_epochs):     # OUTER LOOP (Epochs)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader): # INNER LOOP (Batches)
        
        # --- THE BIG 5 STEPS ---
        
        # Step 1: Zero Gradients
        # Clears old gradients from previous step. Mandatory.
        optimizer.zero_grad()
        
        # Step 2: Forward Pass
        # Compute prediction based on inputs.
        outputs = model(inputs)
        
        # Step 3: Calculate Loss
        # Compare prediction vs truth.
        # NOTE: For BCELoss, targets must be shape (Batch, 1) and float type
        loss = criterion(outputs, targets.float().unsqueeze(1))
        
        # Step 4: Backward Pass (Backprop)
        # Calculate dLoss/dWeights (gradients).
        loss.backward()
        
        # Step 5: Optimizer Step
        # Update weights: W = W - lr * grad
        optimizer.step()
        
    print(f"Epoch {epoch} complete. Loss: {loss.item()}")
```

### 5. Exam Checklist (Common Pitfalls)

1.  **Shape Mismatch:**
    * **Conv to Linear:** Did you flatten correctly? Use `x.view(x.size(0), -1)`.
    * **Loss Input:** Does your target shape match your output shape? (e.g., `(32, 1)` vs `(32)`).
2.  **Zero Grad:** Did you call `optimizer.zero_grad()` *before* `loss.backward()`? If not, gradients accumulate and explode.
3.  **Mode Switching:**
    * `model.train()`: Enables Dropout and BatchNorm updates.
    * `model.eval()`: Disables Dropout/BatchNorm (use during testing/validation).
    * `with torch.no_grad():`: Disables gradient calculation (saves memory during testing).
  
### 6. Complete code example.
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        # Conv2d: in_channels=1 (grayscale), out_channels=8, kernel_size=3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear layer (Fully Connected)
        # Input features must match the flattened output of the conv/pool layers
        # If input is 28x28 -> conv (28x28) -> pool (14x14) -> 8 channels * 14 * 14
        self.fc = nn.Linear(8 * 14 * 14, 10) # 10 output classes

    def forward(self, x):
        # Define the data flow
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        # Flatten the output for the Linear layer (Batch_Size, Features)
        out = out.view(out.size(0), -1) 
        
        out = self.fc(out)
        return out

# 2. Setup Data, Model, Loss, and Optimizer
# Dummy Data: Batch of 5 images, 1 channel, 28x28 pixels
inputs = torch.randn(5, 1, 28, 28) 
# Dummy Targets: 5 labels (integers 0-9)
targets = torch.randint(0, 10, (5,))

model = SimpleCNN()

# Define Loss (Criterion) and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. The Training Step (One Iteration)
print("--- Starting Training Step ---")

# A. Zero the gradients
optimizer.zero_grad() 

# B. Forward Pass
outputs = model(inputs) 

# C. Calculate Loss
loss = criterion(outputs, targets)
print(f"Loss Value: {loss.item()}")

# D. Backward Pass (Backpropagation)
loss.backward()

# E. Optimizer Step (Update Weights)
optimizer.step()

print("--- Step Complete. Weights updated. ---")
```

### Epochs in Practice

Imagine you have:
* A **Training Dataset** of $N = 1000$ images.
* A **Batch Size** of $B = 100$ images.

**Iterations per Epoch:**

$$
\text{Iterations} = \frac{\text{Total Training Examples (N)}}{\text{Batch Size (B)}}
$$

$$
\text{Iterations} = \frac{1000}{100} = 10
$$

This means **10 iterations** (or steps) constitute **1 epoch**.

**What it means:** Over those 10 iterations, every one of the 1000 images will have been used to update the model's weights.

If you train for **10 epochs**, the entire 1000-image dataset will have been processed **10 times** in total.

### Why Use Epochs?

You rarely train a model for just one epoch because the weights often need many exposures to the data to fully converge on an optimal solution.

* **Underfitting (Too few epochs):** If you use too few epochs, the model won't have fully learned the patterns in the data and will perform poorly on both the training and test sets.
* **Overfitting (Too many epochs):** If you use too many epochs, the model might start memorizing the noise and peculiarities of the training data. Its performance on the training set will continue to improve, but its performance on new, unseen data (the test set) will start to degrade.

The optimal number of epochs is usually determined by monitoring the model's performance on a separate **validation set**.
