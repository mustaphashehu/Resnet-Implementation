# ResNet-50 Hand Sign Image Classifier

A deep learning image-classification project that implements **ResNet-50 from scratch with TensorFlow/Keras** and trains it to recognize six classes of hand signs from RGB images.

Rather than importing a prebuilt ResNet model, this project constructs the residual architecture directly using reusable **identity blocks** and **convolutional (projection) blocks**. The notebook covers the complete machine-learning workflow: loading the dataset, preprocessing images and labels, defining residual connections, assembling the full ResNet-50 architecture, training the network, evaluating it on unseen data, and performing inference on individual images.

## Project Results

In the recorded training run, the model achieved:

| Metric | Result |
|---|---:|
| Training samples | 1,080 |
| Test samples | 120 |
| Input image size | 64 × 64 × 3 |
| Number of classes | 6 |
| Training epochs | 10 |
| Batch size | 32 |
| Final training accuracy | 97.96% |
| Test accuracy | **97.50%** |
| Test loss | 0.0985 |

The result shows that a deep residual architecture can learn a strong representation of the six-class hand-sign dataset even with relatively small 64×64 input images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Why ResNet?](#why-resnet)
4. [Dataset](#dataset)
5. [Project Workflow](#project-workflow)
6. [Data Preprocessing](#data-preprocessing)
7. [Residual Learning](#residual-learning)
8. [Identity Block](#identity-block)
9. [Convolutional Block](#convolutional-block)
10. [ResNet-50 Architecture](#resnet-50-architecture)
11. [Training Configuration](#training-configuration)
12. [Training Results](#training-results)
13. [Model Evaluation](#model-evaluation)
14. [Single-Image Inference](#single-image-inference)
15. [Project Structure](#project-structure)
16. [Installation](#installation)
17. [How to Run the Project](#how-to-run-the-project)
18. [Using the Model on Your Own Image](#using-the-model-on-your-own-image)
19. [Reproducibility](#reproducibility)
20. [What This Project Demonstrates](#what-this-project-demonstrates)
21. [Limitations and Possible Improvements](#limitations-and-possible-improvements)
22. [References](#references)

---

## Project Overview

The goal of this project is to build an end-to-end computer vision classifier using the **Residual Network (ResNet)** architecture introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

The model receives a color image with shape:

```text
64 × 64 × 3
```

and produces a probability distribution over six possible hand-sign classes:

```text
Input image
    ↓
Custom ResNet-50
    ↓
6-class Softmax output
    ↓
Predicted class
```

The main focus of the project is not simply using ResNet-50, but understanding and implementing the mechanism that makes residual networks work: **skip connections**.

---

## Problem Statement

Deep convolutional neural networks can learn increasingly complex image features as more layers are added. However, simply making a network deeper does not always make it easier to train. Very deep networks can suffer from optimization difficulties and degradation in performance.

Residual Networks address this problem by allowing information to bypass groups of convolutional layers through shortcut connections.

This project applies that idea to a six-class hand-sign image-classification problem.

For an input image `x`, the network learns to predict one of six output classes.

The final layer therefore contains six neurons with a **softmax** activation:

```text
P(class_0), P(class_1), ..., P(class_5)
```

The predicted class is the class with the highest probability.

---

## Why ResNet?

Traditional deep neural networks attempt to directly learn a transformation:

```text
H(x)
```

A residual block instead learns a residual function:

```text
F(x) = H(x) - x
```

which allows the desired mapping to be written as:

```text
H(x) = F(x) + x
```

The original input `x` is passed around several layers through a shortcut connection and then added back to the transformed output.

Conceptually:

```text
             ┌──────────────────────────────┐
             │                              │
Input x ─────┼──→ Convolutional layers ──→ (+) ──→ ReLU ──→ Output
             │                         ↑    │
             └─────────────────────────┘    │
                      shortcut connection   │
```

This direct path helps information and gradients propagate through deep networks and makes architectures such as ResNet-50 practical to optimize.

---

## Dataset

The project expects the dataset in two **HDF5 (`.h5`) files**:

```text
datasets/
├── train_signs.h5
└── test_signs.h5
```

The notebook loads the following arrays from those files:

### Training file

```text
train_set_x   → training images
train_set_y   → training labels
```

### Test file

```text
test_set_x    → test images
test_set_y    → test labels
list_classes  → available class identifiers
```

The recorded dataset dimensions are:

```text
Training examples: 1080
Test examples:     120

X_train: (1080, 64, 64, 3)
Y_train: (1080, 6)
X_test:  (120, 64, 64, 3)
Y_test:  (120, 6)
```

Each image is therefore a **64×64 RGB image**, and each label is converted into a six-element one-hot encoded vector.

---

## Project Workflow

The full pipeline implemented in the notebook is:

```text
HDF5 dataset
     ↓
Load images and labels
     ↓
Normalize pixel values
     ↓
One-hot encode labels
     ↓
Build residual blocks
     ↓
Assemble ResNet-50
     ↓
Compile model
     ↓
Train for 10 epochs
     ↓
Evaluate on test data
     ↓
Run inference on individual images
```

---

## Data Preprocessing

### Image normalization

Images are originally represented using integer pixel values. The project converts the arrays to `float32` and scales all pixel values into the range `[0, 1]`:

```python
X_train = X_train_orig.astype("float32") / 255.0
X_test = X_test_orig.astype("float32") / 255.0
```

Normalizing the input reduces the scale of the values being passed through the network and provides more suitable numerical inputs for optimization.

### Label encoding

The original class labels are integers. Because the network performs multi-class classification with categorical cross-entropy, labels are converted to one-hot vectors using Keras utilities.

For example, a class label could be represented conceptually as:

```text
Class 2 → [0, 0, 1, 0, 0, 0]
```

The project provides a reusable helper:

```python
def one_hot_encode(labels, num_classes):
    return tf.keras.utils.to_categorical(
        np.asarray(labels).reshape(-1),
        num_classes=num_classes,
    )
```

---

## Residual Learning

The core of the implementation is built around two residual block types:

- **Identity block** — used when the shortcut input already has the same shape as the main path output.
- **Convolutional block** — used when the feature-map dimensions or number of channels must change.

Both blocks use the bottleneck pattern:

```text
1×1 convolution
      ↓
f×f convolution
      ↓
1×1 convolution
```

Batch normalization and ReLU activations are applied throughout the network.

---

## Identity Block

The identity block is used when the dimensions of the input and output are already compatible.

Its main path contains three convolutions:

```text
Input X
   │
   ├────────────────────────────────────────┐
   │                                        │
   ↓                                        │
1×1 Conv                                    │
BatchNorm                                   │
ReLU                                        │
   ↓                                        │
f×f Conv                                    │
BatchNorm                                   │
ReLU                                        │
   ↓                                        │
1×1 Conv                                    │
BatchNorm                                   │
   │                                        │
   └──────────────────────→ Add ←───────────┘
                              ↓
                            ReLU
                              ↓
                            Output
```

The shortcut performs no convolution. The original input is added directly to the main path:

```python
X = Add()([X_shortcut, X])
X = Activation("relu")(X)
```

This is the simplest form of a residual connection.

---

## Convolutional Block

A convolutional block is required when the shortcut and main path do not have matching shapes.

For example, a new ResNet stage may:

- reduce the spatial resolution,
- increase the number of feature channels,
- or do both at the same time.

The shortcut therefore passes through its own `1×1` convolution:

```text
                         Main path
                    1×1 → f×f → 1×1
                   /                 \
Input X ──────────                    (+) → ReLU → Output
                   \                 /
                    1×1 projection
                      Shortcut
```

The shortcut projection uses the same stride required to match the new dimensions:

```python
X_shortcut = Conv2D(
    filters=F3,
    kernel_size=1,
    strides=(s, s),
)(X_shortcut)
```

The projected shortcut can then safely be added to the output of the main branch.

---

## ResNet-50 Architecture

The complete model is assembled using the two residual block functions.

### Network stem

The input first passes through:

```text
Input: 64×64×3
      ↓
Zero padding
      ↓
7×7 convolution, 64 filters, stride 2
      ↓
Batch normalization
      ↓
ReLU
      ↓
3×3 max pooling, stride 2
```

### Residual stages

The remaining architecture contains four residual stages.

| Stage | Structure | Filters |
|---|---|---|
| Stage 2 | 1 convolutional block + 2 identity blocks | `[64, 64, 256]` |
| Stage 3 | 1 convolutional block + 3 identity blocks | `[128, 128, 512]` |
| Stage 4 | 1 convolutional block + 5 identity blocks | `[256, 256, 1024]` |
| Stage 5 | 1 convolutional block + 2 identity blocks | `[512, 512, 2048]` |

Stages 3, 4, and 5 use stride 2 in their opening convolutional block to reduce spatial dimensions while increasing channel depth.

### Classification head

After the final residual stage, the network uses:

```text
Average Pooling
      ↓
Flatten
      ↓
Dense layer with 6 units
      ↓
Softmax
```

The model is created as a Keras Functional API model:

```python
return Model(
    inputs=X_input,
    outputs=X,
    name="resnet50_sign_classifier",
)
```

---

## Training Configuration

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

### Optimizer

**Adam** is used to update the model parameters during training.

### Loss function

**Categorical cross-entropy** is used because the task is a six-class classification problem with one-hot encoded labels.

### Training parameters

```text
Epochs:     10
Batch size: 32
Optimizer:  Adam
Loss:       Categorical cross-entropy
Metric:     Accuracy
```

Training is performed with:

```python
history = model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=32,
)
```

---

## Training Results

The recorded training run produced the following accuracy values:

| Epoch | Training Accuracy |
|---:|---:|
| 1 | 43.89% |
| 2 | 74.63% |
| 3 | 88.89% |
| 4 | 88.06% |
| 5 | 94.17% |
| 6 | 83.70% |
| 7 | 91.11% |
| 8 | 93.15% |
| 9 | 95.74% |
| 10 | **97.96%** |

The final recorded training loss was:

```text
0.0862
```

Training was not perfectly monotonic—for example, accuracy dropped during epoch 6—but the model recovered and finished with high training accuracy.

---

## Model Evaluation

After training, the model is evaluated on the 120-image test set:

```python
test_loss, test_accuracy = model.evaluate(
    X_test,
    Y_test,
    verbose=1,
)
```

Recorded evaluation result:

```text
Test loss:     0.0985
Test accuracy: 97.5000%
```

So the final model correctly classified approximately **97.5% of the held-out test examples** in the recorded run.

---

## Single-Image Inference

The notebook also includes a reusable `predict_image()` function for testing the model on an individual image.

The function performs the same preprocessing used during training:

1. Load the image.
2. Resize it to `64×64`.
3. Convert it to a NumPy array.
4. Add a batch dimension.
5. Normalize pixel values by dividing by 255.
6. Run the model.
7. Select the class with the highest softmax probability.
8. Display the image and predicted class.

Example:

```python
predicted_class, probabilities = predict_image(
    model,
    "images/my_image.jpg",
)

print("Predicted class:", predicted_class)
print("Class probabilities:", probabilities)
```

The returned `probabilities` array contains the model's probability for each of the six classes.

---

## Project Structure

A simple repository layout for the project is:

```text
resnet50-hand-sign-classifier/
│
├── README.md
├── ResNet50_Hand_Sign_Classifier_Project.ipynb
│
├── datasets/
│   ├── train_signs.h5
│   └── test_signs.h5
│
└── images/
    └── my_image.jpg        # optional image for inference
```

The notebook expects the dataset directory to be named `datasets` unless a different path is explicitly passed to `load_signs_dataset()`.

---

## Installation

### 1. Clone or download the repository

```bash
git clone <your-repository-url>
cd resnet50-hand-sign-classifier
```

If you are not using Git, download the project files and open a terminal inside the project directory.

### 2. Create a virtual environment

Using Python's built-in `venv`:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install the required Python packages

The notebook uses the following main dependencies:

```text
TensorFlow
NumPy
h5py
Matplotlib
Jupyter
```

They can be installed with:

```bash
pip install tensorflow numpy h5py matplotlib jupyter
```

The exact TensorFlow version is not pinned in the notebook, so environment-specific compatibility may vary.

---

## How to Run the Project

### Step 1 — Add the dataset

Place the two required HDF5 files inside:

```text
datasets/train_signs.h5
datasets/test_signs.h5
```

### Step 2 — Start Jupyter

```bash
jupyter notebook
```

### Step 3 — Open the notebook

Open:

```text
ResNet50_Hand_Sign_Classifier_Project.ipynb
```

### Step 4 — Run the notebook from top to bottom

The notebook will:

```text
Import dependencies
       ↓
Load the HDF5 dataset
       ↓
Define identity_block()
       ↓
Define convolutional_block()
       ↓
Build ResNet-50
       ↓
Normalize images
       ↓
One-hot encode labels
       ↓
Compile the network
       ↓
Train for 10 epochs
       ↓
Evaluate the test set
       ↓
Make optional single-image predictions
```

---

## Using the Model on Your Own Image

Place an image inside the project, for example:

```text
images/my_image.jpg
```

Then run:

```python
predicted_class, probabilities = predict_image(
    model,
    "images/my_image.jpg",
)
```

The function automatically resizes the image to the network's expected `64×64` resolution.

For the most meaningful results, the image should resemble the type of hand-sign images represented in the training dataset.

---

## Reproducibility

The notebook sets random seeds for NumPy and TensorFlow:

```python
np.random.seed(1)
tf.random.set_seed(1)
```

The custom convolutional layers also use explicitly seeded initializers such as:

```python
random_uniform(seed=0)
```

and:

```python
glorot_uniform(seed=0)
```

These settings help make experiments more reproducible, although exact numerical results can still vary between TensorFlow versions, hardware configurations, backend implementations, and execution environments.

---

## What This Project Demonstrates

This project demonstrates practical experience with several core deep-learning and software-development concepts:

- Building a deep convolutional neural network with the **Keras Functional API**.
- Implementing **ResNet-50 architecture manually** instead of relying on a prebuilt application model.
- Understanding and implementing **skip/residual connections**.
- Designing reusable **identity and convolutional blocks**.
- Using `1×1`, `3×3`, and `7×7` convolutions for different roles in a CNN architecture.
- Applying **batch normalization** and **ReLU** activations throughout a deep network.
- Handling dimension changes using projection shortcuts.
- Loading structured image datasets from **HDF5 files**.
- Normalizing image data for neural-network training.
- Performing **one-hot encoding** for multi-class targets.
- Training with **Adam** and **categorical cross-entropy**.
- Evaluating generalization on a held-out test set.
- Implementing an end-to-end inference function for new images.
- Structuring a deep-learning experiment in a reproducible Jupyter notebook.

---

## Limitations and Possible Improvements

The current project provides a complete working ResNet-50 classification pipeline, but several extensions could make it stronger.

### Data augmentation

The current notebook normalizes the training images but does not apply augmentation. Transformations such as small rotations, translations, zoom, or brightness changes could improve robustness if they are appropriate for the dataset.

### Validation set

The recorded training call uses the training data directly without a separate validation split. Adding a validation set would make it easier to monitor generalization during training and identify overfitting before evaluating on the final test set.

### Training callbacks

Useful additions could include:

```text
EarlyStopping
ModelCheckpoint
ReduceLROnPlateau
```

These can improve the training workflow and preserve the best-performing model automatically.

### Saved model artifacts

The notebook currently demonstrates training and inference in the same session. A production-oriented extension could save the trained network after training and load it later without retraining.

For example:

```python
model.save("resnet50_sign_classifier.keras")
```

### Additional evaluation metrics

Overall accuracy is useful, but further analysis could include:

- confusion matrix,
- per-class precision,
- per-class recall,
- F1 score,
- misclassified image inspection.

These would show which classes are easiest or most difficult for the model to distinguish.

### Training-history visualization

The `history` object returned by `model.fit()` can be used to visualize training loss and accuracy across epochs. This would make optimization behavior easier to interpret.

### Hyperparameter experimentation

Possible experiments include changing:

- learning rate,
- batch size,
- number of epochs,
- optimizer,
- augmentation strategy,
- initialization strategy.

---

## References

The implementation is based on the residual-learning architecture introduced in:

**Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun — _Deep Residual Learning for Image Recognition_ (2015)**  
https://arxiv.org/abs/1512.03385

Additional architecture and API references:

- TensorFlow / Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras
- François Chollet's early Keras ResNet implementation: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

---

## Summary

This project implements a complete **ResNet-50 image-classification system from first principles using TensorFlow/Keras**. The network is constructed from custom residual blocks, trained on 1,080 hand-sign images, evaluated on 120 unseen images, and extended with a utility for predicting individual images.

The recorded model achieved **97.5% test accuracy after 10 epochs**, demonstrating both the effectiveness of residual learning and the ability of the implementation to train a deep convolutional architecture successfully on the six-class dataset.
