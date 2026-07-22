#pragma once

#include "Matrix.h"
#include "Random.h"
#include "Arena.h"
#include "Types.h"

typedef struct
{
    u32 layers;

    Matrix **weights;
    Matrix **biases;

} NeuralNet;


// Create Network

NeuralNet *NN_Create(
    Arena *arena,
    u32 input,
    u32 hidden,
    u32 output,
    pcg32_random_t *rng
);


// Forward

Matrix *NN_Forward(
    NeuralNet *nn,
    Matrix *input,
    Arena *arena
);


// Loss

f32 NN_Loss(
    Matrix *prediction,
    Matrix *target
);


// Training

void NN_Train(
    NeuralNet *nn,
    Matrix *input,
    Matrix *target,
    f32 learning_rate,
    Arena *arena
);