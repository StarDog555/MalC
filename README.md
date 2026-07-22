# Malc Neural Network Library

A neural network library written completely in C.

Malc is a lightweight machine learning framework built from scratch. It includes:

* Custom arena memory allocator
* Matrix mathematics
* Neural network creation
* Forward propagation
* Backpropagation
* Gradient descent
* ReLU activation
* Softmax output
* Cross entropy loss
* Random weight initialization

The goal of this project is to understand and build neural networks at a low level without external machine learning libraries.

---

# Table of Contents

1. Installation
2. Project Structure
3. Building
4. Creating an Arena
5. Creating a Neural Network
6. Input and Target Data
7. Forward Pass
8. Loss Calculation
9. Training
10. Matrix Library
11. Example Program
12. Troubleshooting
13. Future Improvements

---

# Installation

## Requirements

### Windows

Required:

* Visual Studio Build Tools
* MSVC Compiler

Open:

Developer Command Prompt for Visual Studio

### Linux

Required:

* GCC or Clang
* Math library

---

# Project Structure

```
Malc.C

├── include
│
│   ├── Malc.h
│   ├── Matrix.h
│   ├── NeuralNet.h
│   ├── Arena.h
│   ├── Types.h
│   └── random.h
│
├── src
│
│   ├── main.c
│   ├── Matrix.c
│   ├── NeuralNet.c
│   ├── Arena.c
│   └── random.c
│
└── README.md
```

---

# Building

## Windows

Navigate into the source folder:

```
cd src
```

Compile:

```
cl main.c Matrix.c Arena.c random.c NeuralNet.c /I ..\include /Fe:malc.exe
```

Run:

```
malc.exe
```

---

## Linux

Compile:

```
gcc main.c Matrix.c Arena.c random.c NeuralNet.c -I ../include -lm -o malc
```

Run:

```
./malc
```

---

# Memory System

Malc uses a custom arena allocator.

Instead of using:

```
malloc()
free()
```

Malc allocates memory from a large block.

Example:

```c
Arena arena;

Arena_Create(
    &arena,
    16 * 1024 * 1024
);
```

This creates a 16 MB memory arena.

All matrices and neural network data are stored inside this arena.

When finished:

```c
Arena_Destroy(&arena);
```

Everything is freed at once.

---

# Creating A Neural Network

Include the library:

```c
#include "../include/Malc.h"
```

Create memory:

```c
Arena arena;

Arena_Create(
    &arena,
    16 * 1024 * 1024
);
```

Create random generator:

```c
pcg32_random_t rng;

PCG32_Seed(
    &rng,
    1234
);
```

Create network:

```c
NeuralNet *nn =
    NN_Create(
        &arena,
        1,
        32,
        3,
        &rng
    );
```

The parameters are:

```
NN_Create(
    arena,
    input nodes,
    hidden nodes,
    output nodes,
    random generator
)
```

Example:

```
Input Layer:
1 neuron

Hidden Layer:
32 neurons

Output Layer:
3 classes
```

---

# Creating Input And Target

Create input:

```c
Matrix *input =
    Mat_Create(
        &arena,
        1,
        1
    );
```

Example:

```
Input:

1.5
```

Create target:

```c
Matrix *target =
    Mat_Create(
        &arena,
        3,
        1
    );
```

Targets use one-hot encoding.

Example:

Class 0:

```
1
0
0
```

Class 1:

```
0
1
0
```

Class 2:

```
0
0
1
```

Example:

```c
Mat_Clear(target);

target->data[1] = 1.0f;
```

This selects class 1.

---

# Forward Pass

The forward pass runs the input through the network.

Example:

```c
Matrix *prediction =
    NN_Forward(
        nn,
        input,
        &arena
    );
```

Example output:

```
0.001
0.998
0.001
```

Meaning:

```
Class 0:
0.1%

Class 1:
99.8%

Class 2:
0.1%
```

The highest value is the prediction.

---

# Loss Function

Loss measures how incorrect the prediction is.

Example:

```c
float loss =
    NN_Loss(
        prediction,
        target
    );
```

Correct prediction:

```
Prediction:

0
1
0


Target:

0
1
0


Loss:

0.00000
```

Wrong prediction:

```
Prediction:

1
0
0


Target:

0
0
1


Loss:

16.11810
```

Lower loss means the model is improving.

---

# Training

Training updates the network weights.

Example:

```c
NN_Train(
    nn,
    input,
    target,
    0.001f,
    &arena
);
```

A normal training loop:

```c
for(int epoch = 0;
    epoch < 100000;
    epoch++)
{
    NN_Train(
        nn,
        input,
        target,
        0.001f,
        &arena
    );
}
```

---

# Example Dataset

The network can learn:

```
Input       Class

0.5           0

1.5           1

2.5           2
```

After training:

Input:

```
0.5
```

Output:

```
1.000
0.000
0.000
```

---

Input:

```
1.5
```

Output:

```
0.000
1.000
0.000
```

---

Input:

```
2.5
```

Output:

```
0.000
0.000
1.000
```

---

# Matrix Library

## Create Matrix

```c
Matrix *matrix =
    Mat_Create(
        &arena,
        rows,
        cols
    );
```

---

## Clear Matrix

```c
Mat_Clear(matrix);
```

Sets all values to zero.

---

## Fill Matrix

```c
Mat_Fill(
    matrix,
    value
);
```

Example:

```c
Mat_Fill(matrix, 1.0f);
```

---

## Scale Matrix

```c
Mat_Scale(
    matrix,
    scale
);
```

Example:

```
[1 2 3]

scale 2

[2 4 6]
```

---

## Add

```c
Mat_Add(
    output,
    a,
    b
);
```

Formula:

```
output = a + b
```

---

## Subtract

```c
Mat_Subtract(
    output,
    a,
    b
);
```

Formula:

```
output = a - b
```

---

## Matrix Multiply

```c
Mat_Multiply(
    output,
    a,
    b
);
```

Formula:

```
output = a * b
```

---

## ReLU

```c
Mat_Relu(
    output,
    input
);
```

Example:

Input:

```
-5
2
-1
```

Output:

```
0
2
0
```

---

## Softmax

```c
Mat_Softmax(
    output,
    input
);
```

Converts values into probabilities.

Example:

Input:

```
5
2
1
```

Output:

```
0.94
0.05
0.01
```

---

# Gradient Operations

## Hadamard Product

Element-wise multiplication.

```c
Mat_Hadamard(
    output,
    a,
    b
);
```

Example:

```
[1 2 3]

*

[4 5 6]


=

[4 10 18]
```

---

## Outer Product

```c
Mat_Outer(
    output,
    a,
    b
);
```

Used to calculate neural network weight gradients.

---

## Add Scaled

```c
Mat_AddScaled(
    matrix,
    gradient,
    scale
);
```

Used during learning:

```
weights =
weights - learning_rate * gradient
```

---

# How Training Works

Every training step:

```
Input

 |

Forward Pass

 |

Prediction

 |

Loss

 |

Calculate Gradients

 |

Backpropagation

 |

Update Weights

 |

Repeat
```

---

# Current Features

Implemented:

* Matrix operations
* Arena memory management
* Fully connected neural networks
* ReLU activation
* Softmax output
* Cross entropy loss
* Backpropagation
* Gradient descent

---

# Current Limitations

Currently:

* One hidden layer
* CPU only
* No batch processing
* No model saving

---

# Future Improvements

Planned:

* Multiple hidden layers
* Mini batch training
* Adam optimizer
* Momentum
* Dropout
* Model save/load
* GPU acceleration
* Convolution layers

---

# Troubleshooting

## Prediction is always one class

Try:

* More epochs
* Lower learning rate
* More hidden neurons
* Better training examples

---

## Loss stays high

Check:

* Target labels
* Input values
* Learning rate
* Network size

---

## Out of memory

Increase:

```c
Arena_Create(
    &arena,
    larger_size
);
```

Example:

```c
Arena_Create(
    &arena,
    64 * 1024 * 1024
);
```

---

# License

Free to use, modify, and learn from.

---

# Author

Malc Neural Network Library

A pure C neural network implementation built for learning how machine learning works internally.

