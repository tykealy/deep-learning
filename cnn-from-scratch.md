# Deep Learning Practice — PyTorch to Object Re-Identification

> **Primary goal:** Build a strong understanding of deep learning using pure PyTorch, beginning with image classification, progressing through model training and evaluation, and eventually moving toward Object Re-Identification (Re-ID).
>
> **Hardware:** MacBook Pro M3 Pro (Apple Silicon / MPS)
>
> **Environment:** Conda, Python, PyTorch, TorchVision
>
> **First dataset:** CIFAR-10 (60,000 images, 10 classes)
>
> **Research direction:** Object Re-Identification, particularly cross-view and aerial imagery applications.

---

## 1. Learning Objectives

By completing this plan, I should be able to:

- Understand the fundamental mathematics of neural networks.
- Implement and train neural networks using pure PyTorch.
- Understand how CNNs extract features from images.
- Explain forward propagation and backpropagation.
- Understand how loss functions and optimizers work.
- Perform training, validation, and testing correctly.
- Analyze accuracy, loss curves, and overfitting.
- Understand how architecture choices affect accuracy and efficiency.
- Extract feature embeddings from trained neural networks.
- Develop a foundation for object detection and Re-ID research.
- Read and reproduce basic experiments from deep learning papers.

## 2. Learning Roadmap

| Phase | Topic | Expected Outcome |
|---|---|---|
| 0 | PyTorch fundamentals | Understand tensors, modules, and autograd |
| 1 | Dataset preparation | Load and inspect CIFAR-10 |
| 2 | CNN architecture | Build a CNN from scratch |
| 3 | Training fundamentals | Implement a complete training loop |
| 4 | Training experiments | Understand hyperparameters and optimization |
| 5 | Model evaluation | Evaluate accuracy and generalization |
| 6 | Model improvements | Experiment with architecture and regularization |
| 7 | Feature extraction | Extract and compare learned embeddings |
| 8 | Object detection | Apply training knowledge to YOLO |
| 9 | Object Re-ID | Build a basic feature-matching system |
| 10 | Research preparation | Read papers and reproduce baselines |

---

# Phase 0 — PyTorch Fundamentals

**Objective:** Understand PyTorch operations before working with a complete training pipeline.

### Topics

- Tensors and tensor dimensions
- Tensor indexing, slicing, and broadcasting
- Tensor operations and matrix multiplication
- `torch.Tensor` and `torch.nn.Parameter`
- `nn.Module` and `forward()`
- `nn.Linear` and `nn.Conv2d`
- Activation functions: ReLU, sigmoid, softmax
- Automatic differentiation with autograd
- Computational graphs
- CPU, CUDA, and Apple MPS devices

### Practical Exercises

- [ ] Create tensors with different dimensions.
- [ ] Perform matrix multiplication.
- [ ] Implement a basic linear layer.
- [ ] Calculate a simple function's gradient using autograd.
- [ ] Compare calculated gradients with manually derived gradients.
- [ ] Move a tensor to MPS and execute operations.
- [ ] Inspect the weights and biases of an `nn.Linear` layer.

### Milestone

Explain what happens when executing:

```python
y = model(x)
loss = criterion(y, target)
loss.backward()
```

Understand why gradients are calculated and where they are stored.

---

# Phase 1 — Dataset Preparation

**Dataset: CIFAR-10**

| Property | Value |
|---|---|
| Total images | 60,000 |
| Training pool | 50,000 |
| Training subset | 45,000 |
| Validation subset | 5,000 |
| Test set | 10,000 |
| Image size | 32 × 32 |
| Channels | 3 (RGB) |
| Classes | 10 |

### Topics

- Supervised learning
- Image tensors
- Labels and class indices
- Training, validation, and test splits
- `torch.utils.data.Dataset`
- `DataLoader`
- Batches, iterations, and epochs
- Image normalization
- Data augmentation
- Data leakage

### Practical Exercises

- [ ] Download CIFAR-10 through TorchVision.
- [ ] Inspect the structure of a dataset sample.
- [ ] Print an image tensor's shape.
- [ ] Display sample images with their labels.
- [ ] Split the training pool into training and validation sets.
- [ ] Create DataLoaders with batch size 128.
- [ ] Apply normalization.
- [ ] Apply random horizontal flipping and cropping only to training data.
- [ ] Understand how shuffling changes the order of training samples.

### Questions to Answer

1. Why do we need batches instead of using all 45,000 images simultaneously?
2. What is the difference between an iteration and an epoch?
3. Why should validation data not be used for gradient updates?
4. Why do we normalize images?
5. What is the purpose of data augmentation?

### Milestone

Successfully load batches with shape:

```text
Images: (128, 3, 32, 32)
Labels: (128,)
```

---

# Phase 2 — Build a CNN From Scratch

**Objective:** Understand how image features are learned and transformed through neural network layers.

### Proposed Architecture

```text
Input Image
(3, 32, 32)
       |
Conv2d(3, 32, kernel=3, padding=1)
ReLU
       |
Conv2d(32, 32, kernel=3, padding=1)
ReLU
MaxPool2d(2)
       |
Conv2d(32, 64, kernel=3, padding=1)
ReLU
MaxPool2d(2)
       |
Conv2d(64, 128, kernel=3, padding=1)
ReLU
       |
AdaptiveAvgPool2d(1)
       |
Flatten
       |
Linear(128, 10)
       |
10 Class Logits
```

### Topics

- Convolutional layers
- Kernels, channels, stride, and padding
- Feature maps
- Pooling
- Receptive fields
- Activation functions
- Flattening
- Fully connected layers
- Global average pooling
- Trainable parameters

### Practical Exercises

- [ ] Implement the CNN using `nn.Module`.
- [ ] Define `__init__()` and `forward()`.
- [ ] Print the model architecture.
- [ ] Calculate the number of trainable parameters.
- [ ] Print tensor shapes after each layer.
- [ ] Understand how the spatial resolution changes.
- [ ] Explain why the final output contains 10 values.

### Expected Shapes

| Layer | Output shape (excluding batch) |
|---|---|
| Input | (3, 32, 32) |
| Conv1 | (32, 32, 32) |
| Conv2 | (32, 32, 32) |
| Pool1 | (32, 16, 16) |
| Conv3 | (64, 16, 16) |
| Pool2 | (64, 8, 8) |
| Conv4 | (128, 8, 8) |
| Global average pooling | (128, 1, 1) |
| Flatten | (128,) |
| Linear | (10,) |

**Expected trainable parameters:** 103,786, with default biases.

### Milestone

Explain how an image tensor of shape `(3, 32, 32)` becomes a 10-dimensional output vector.

---

# Phase 3 — Implement the Training Loop

**Objective:** Understand how a neural network learns from labeled data.

### Training Pipeline

```text
Input Batch
    |
Forward Propagation
    |
Predicted Class Logits
    |
Calculate Loss
    |
Backpropagation
    |
Calculate Parameter Gradients
    |
Optimizer Updates Weights
    |
Next Batch
```

### Topics

#### 3.1 Forward Propagation

- How input tensors pass through layers
- Matrix multiplication and convolution
- Activation functions
- Logits versus probabilities
- How output dimensions are determined

#### 3.2 Loss Functions

- Cross-entropy loss
- Softmax
- Logits
- Class indices versus one-hot labels
- Batch loss reduction
- Why lower loss generally indicates better fitting

#### 3.3 Backpropagation

- Computational graphs
- Chain rule
- Partial derivatives
- Gradients of weights and biases
- Gradient accumulation
- Automatic differentiation

#### 3.4 Optimization

- Gradient descent
- Stochastic gradient descent (SGD)
- Adam optimizer
- Learning rate
- Weight updates
- Optimizer state

### Training Configuration

```python
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
```

### Practical Exercises

- [ ] Implement `model.train()`.
- [ ] Iterate through training batches.
- [ ] Perform forward propagation.
- [ ] Calculate cross-entropy loss.
- [ ] Call `optimizer.zero_grad()`.
- [ ] Call `loss.backward()`.
- [ ] Inspect one parameter's gradient.
- [ ] Call `optimizer.step()`.
- [ ] Compare weights before and after an update.
- [ ] Calculate average training loss.
- [ ] Calculate training accuracy.

### Important Experiment

Inspect one training iteration:

```python
parameter = model.classifier.weight

before = parameter.detach().clone()

optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

print("Gradient:", parameter.grad)
print("Gradient shape:", parameter.grad.shape)

optimizer.step()

after = parameter.detach().clone()

print("Weight change:", (after - before).abs().mean())
```

### Questions to Answer

1. How does cross-entropy calculate prediction error?
2. Why does `backward()` not update weights directly?
3. Why must gradients be cleared?
4. Does each weight have its own gradient?
5. How does Adam differ from ordinary gradient descent?
6. Why can a neural network have decreasing loss but imperfect accuracy?

### Milestone

Train the CNN for 10 epochs and explain every important operation in the training loop.

---

# Phase 4 — Training Experiments

**Objective:** Understand how hyperparameters affect model learning, convergence, and computational cost.

Change one variable at a time while keeping the other settings consistent.

## Experiment 1 — Batch Size

Compare:

| Run | Batch size |
|---|---:|
| A | 32 |
| B | 64 |
| C | 128 |
| D | 256 |

Observe:

- Loss behavior
- Training accuracy
- Validation accuracy
- Time per epoch
- Memory usage

## Experiment 2 — Learning Rate

Compare:

| Run | Learning rate |
|---|---:|
| A | 0.01 |
| B | 0.001 |
| C | 0.0001 |

Observe convergence speed and training stability.

## Experiment 3 — Optimizers

Compare Adam and SGD with momentum.

Understand why optimizer hyperparameters may require separate tuning.

## Experiment 4 — Number of Epochs

Compare 5, 10, and 20 epochs.

Observe whether validation accuracy improves, plateaus, or deteriorates.

## Experiment 5 — Data Augmentation

Compare training with and without random crop and horizontal flip.

### Experiment Tracking

Record results in a table:

| Experiment | Epochs | Batch | LR | Optimizer | Train Acc | Val Acc | Epoch Time |
|---|---|---|---|---|---|---|---|
| Baseline | 10 | 128 | 0.001 | Adam | TBD | TBD | TBD |
| Batch 64 | 10 | 64 | 0.001 | Adam | TBD | TBD | TBD |
| LR 0.0001 | 10 | 128 | 0.0001 | Adam | TBD | TBD | TBD |

Keep a fixed train/validation split and random seed for comparisons, and repeat important experiments with multiple seeds when possible.

### Milestone

Identify a training configuration that provides reasonable validation accuracy and explain why it was selected.

---

# Phase 5 — Model Evaluation

**Objective:** Evaluate model quality using data not used for gradient updates.

### Topics

- Training accuracy versus validation accuracy
- Test-set evaluation
- Confusion matrices
- Precision
- Recall
- F1-score
- Per-class accuracy
- Overfitting and underfitting
- Generalization
- Loss curves
- Inference latency

### Practical Exercises

- [ ] Implement validation using `model.eval()`.
- [ ] Use `torch.inference_mode()`.
- [ ] Calculate accuracy on the validation set.
- [ ] Save the best model checkpoint.
- [ ] Plot training loss by epoch.
- [ ] Plot training and validation accuracy.
- [ ] Create a confusion matrix.
- [ ] Examine incorrectly classified images.
- [ ] Evaluate the selected model on the test set once experiments are complete.
- [ ] Measure inference time on MPS.

### Questions to Answer

1. Why can training accuracy exceed validation accuracy?
2. How can a confusion matrix reveal model weaknesses?
3. What is the difference between precision and recall?
4. How do we recognize overfitting?
5. Why is test accuracy not necessarily representative of real-world performance?
6. How does model complexity affect inference speed?

### Milestone

Produce a short evaluation report containing:

- Final training configuration
- Training and validation curves
- Best validation accuracy
- Final test accuracy
- Confusion matrix
- Examples of incorrect predictions
- Observed limitations

---

# Phase 6 — Improve the CNN

**Objective:** Understand architectural design decisions and their effect on performance.

### Experiments

- [ ] Increase the number of convolutional channels.
- [ ] Add an additional convolutional layer.
- [ ] Compare ReLU and LeakyReLU.
- [ ] Add batch normalization.
- [ ] Experiment with dropout.
- [ ] Try weight decay.
- [ ] Compare max pooling and average pooling.
- [ ] Investigate how changing feature-map resolution affects performance.
- [ ] Compare parameter count and inference latency.

### Research Questions

- Does a deeper network necessarily perform better?
- Why do skip connections help train deep networks?
- What is the relationship between receptive field and image recognition?
- Does increasing model size always improve generalization?
- Can a smaller model achieve similar accuracy with lower computational cost?

### Milestone

Compare at least two CNN architectures and explain their performance differences.

---

# Phase 7 — Learn Feature Extraction and Embeddings

**Objective:** Transition from classification to concepts directly relevant to Object Re-ID.**

Instead of asking:

> What class does this image belong to?

We begin asking:

> How can a neural network represent an image so that similar objects have similar feature vectors?

### Topics

- Feature extraction
- Representation learning
- Embedding vectors
- Global average pooling
- Feature normalization
- Euclidean distance
- Cosine similarity
- Nearest-neighbor retrieval
- Feature-space visualization

### Practical Exercise

Use the trained CNN as a feature extractor.

```python
model.eval()

with torch.inference_mode():
    features = model.features(images)
    embeddings = torch.flatten(features, 1)

print(embeddings.shape)
# (batch_size, 128)
```

Compare embeddings:

```python
import torch.nn.functional as F

similarity = F.cosine_similarity(
    embeddings[0].unsqueeze(0),
    embeddings[1].unsqueeze(0)
)

print(similarity.item())
```

### Experiments

- [ ] Extract embeddings for validation images.
- [ ] Normalize feature vectors.
- [ ] Compare cosine similarity between images.
- [ ] Retrieve the nearest images to a selected query image.
- [ ] Visualize embeddings using PCA.
- [ ] Investigate whether images from the same class cluster together.
- [ ] Examine cases where visually similar images belong to different classes.

### Important Limitation

CIFAR-10 provides category labels, not unique object identities.

Therefore, same-class similarity is **not equivalent to Object Re-ID**. This phase only introduces representation learning and retrieval.

### Milestone

Build a basic image-retrieval application using features extracted from the trained CNN.

---

# Phase 8 — Return to Object Detection

**Objective:** Apply the deep learning knowledge to the Ultralytics project.**

### Topics

- Detection versus classification
- Bounding-box regression
- Multi-scale feature maps
- Backbone, neck, and detection head
- Classification and box-regression branches
- Ground-truth assignment
- IoU-based losses
- Confidence filtering
- Detection evaluation: precision, recall, mAP
- Inference architecture and postprocessing

### Practical Exercises

- [ ] Run pretrained YOLO inference.
- [ ] Inspect its raw prediction tensors.
- [ ] Inspect bounding-box and class branches.
- [ ] Fine-tune YOLO on a small dataset.
- [ ] Read and interpret training logs.
- [ ] Evaluate detection performance.
- [ ] Inspect missed detections and false positives.
- [ ] Compare inference results across input resolutions.
- [ ] Experiment with object crops extracted from detections.

### Milestone

Train, evaluate, and explain a complete object-detection pipeline.

Understand how its detected objects can be passed into a separate Re-ID model.

---

# Phase 9 — Object Re-Identification Fundamentals

**Objective:** Move from category classification toward identity matching across images or viewpoints.

### Topics

- Object identity versus object category
- Query and gallery images
- Embedding networks
- Metric learning
- Contrastive loss
- Triplet loss
- Identity classification loss
- Hard negative mining
- Cross-view generalization
- Domain shift
- Rank-1 accuracy
- Re-ID mean Average Precision
- Retrieval and nearest-neighbor search

### Practical Exercises

- [ ] Select a suitable identity-labeled Re-ID dataset.
- [ ] Understand query/gallery dataset organization.
- [ ] Load images with identity labels.
- [ ] Create an embedding network.
- [ ] Compare embeddings using cosine similarity.
- [ ] Implement a basic triplet-loss experiment.
- [ ] Create a query-to-gallery retrieval pipeline.
- [ ] Evaluate Rank-1 accuracy and mAP.
- [ ] Analyze successful and unsuccessful matches.

### Research Considerations

For aerial or cross-view Re-ID, investigate:

- Viewpoint differences
- Scale variation
- Occlusion
- Lighting changes
- Background interference
- Motion and camera geometry
- Domain adaptation
- Identity-discriminative feature learning

### Milestone

Implement a simple Re-ID baseline that retrieves images of the same object identity from a gallery.

---

# Phase 10 — Research Preparation

**Objective:** Develop the ability to understand and reproduce research papers.

### Sources

- [CVF Open Access](https://openaccess.thecvf.com/menu)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- [Hung-yi Lee Deep Learning Tutorial](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_Deep.pptx)

### Paper Reading Checklist

For each paper, identify:

- [ ] Research problem
- [ ] Motivation
- [ ] Related work
- [ ] Proposed method
- [ ] Network architecture
- [ ] Training objectives and loss functions
- [ ] Datasets
- [ ] Evaluation metrics
- [ ] Experimental results
- [ ] Ablation studies
- [ ] Limitations
- [ ] Potential improvements

### Reproduction Workflow

1. Understand the research problem.
2. Read the architecture and loss-function sections.
3. Identify the datasets and evaluation protocol.
4. Find an official implementation, if available.
5. Reproduce a baseline.
6. Compare results with the paper.
7. Identify possible reasons for differences.
8. Experiment with one controlled modification.
9. Record findings and limitations.

### Milestone

Reproduce one Re-ID baseline and produce a short technical report explaining its architecture, training, and evaluation.

---

# Final Deliverables

By the end of this learning plan, aim to have a GitHub repository containing:

```text
deep-learning-practice/
│
├── README.md
├── LEARNING_PLAN.md
├── environment.yml
│
├── 01-pytorch-basics/
│   ├── tensors.py
│   ├── autograd.py
│   └── linear_model.py
│
├── 02-cifar10-cnn/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── checkpoints/
│
├── 03-experiments/
│   ├── results.csv
│   └── notebooks/
│
├── 04-feature-extraction/
│   ├── extract.py
│   └── retrieval.py
│
├── 05-object-detection/
│   ├── train.py
│   └── evaluate.py
│
├── 06-object-reid/
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
└── reports/
    ├── cifar10-results.md
    ├── architecture-comparison.md
    └── reid-baseline.md
```

Keep downloaded datasets and large model checkpoints out of Git using `.gitignore`. Track reproducible configurations, source code, metrics, and documentation.

## Definition of Completion

This plan is complete when I can independently:

1. Design a CNN architecture in PyTorch.
2. Build a dataset and DataLoader pipeline.
3. Train a network using a manual training loop.
4. Explain gradients, backpropagation, and optimization.
5. Evaluate model performance correctly.
6. Compare architectural and hyperparameter choices.
7. Extract learned feature embeddings.
8. Explain how detection and Re-ID differ.
9. Train and evaluate an identity-matching baseline.
10. Understand and reproduce the main experiments of a relevant research paper.

---

## Current Progress

- [x] Load and run pretrained YOLO inference.
- [x] Understand bounding-box and class prediction outputs.
- [x] Study convolutional filters and output-channel dimensions.
- [x] Understand how prediction branches preserve spatial alignment.
- [ ] Build and train the CIFAR-10 CNN.
- [ ] Understand the complete training loop.
- [ ] Evaluate the trained CNN.
- [ ] Perform controlled experiments.
- [ ] Extract image feature embeddings.
- [ ] Return to YOLO training.
- [ ] Begin Object Re-ID experiments.

**Immediate next task:** Complete the baseline CIFAR-10 training run using pure PyTorch, record training and validation accuracy, and inspect how the model's weights change after backpropagation.