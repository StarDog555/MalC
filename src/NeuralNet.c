#include <stdio.h>
#include <math.h>

#include "../include/NeuralNet.h"

NeuralNet *NN_Create(Arena *arena, u32 input, u32 hidden, u32 output, pcg32_random_t *rng)
{
    if (arena == NULL)
        return NULL;

    NeuralNet *nn = Arena_Alloc(arena, sizeof(NeuralNet));

    if (nn == NULL)
        return NULL;

    nn->layers = 2;

    nn->weights = Arena_Alloc(arena, 2 * sizeof(Matrix *));
    nn->biases = Arena_Alloc(arena, 2 * sizeof(Matrix *));

    if (nn->weights == NULL || nn->biases == NULL)
        return NULL;


    nn->weights[0] = Mat_Create(arena, hidden, input);
    nn->biases[0] = Mat_Create(arena, hidden, 1);

    nn->weights[1] = Mat_Create(arena, output, hidden);
    nn->biases[1] = Mat_Create(arena, output, 1);

    if (nn->weights[0] == NULL ||
        nn->weights[1] == NULL)
        return NULL;


    Mat_Xavier_Init(nn->weights[0], rng, input, hidden);
    Mat_Xavier_Init(nn->weights[1], rng, hidden, output);

    Mat_Clear(nn->biases[0]);
    Mat_Clear(nn->biases[1]);

    return nn;
}


Matrix *NN_Forward(NeuralNet *nn, Matrix *input, Arena *arena)
{
    if (nn == NULL || input == NULL)
        return NULL;


    Matrix *hidden = Mat_Create(
        arena,
        nn->weights[0]->rows,
        1
    );

    if (hidden == NULL)
        return NULL;


    Mat_Multiply(hidden, nn->weights[0], input);
    Mat_Add(hidden, hidden, nn->biases[0]);
    Mat_Relu(hidden, hidden);



    Matrix *output = Mat_Create(
        arena,
        nn->weights[1]->rows,
        1
    );

    if (output == NULL)
        return NULL;


    Mat_Multiply(output, nn->weights[1], hidden);
    Mat_Add(output, output, nn->biases[1]);
    Mat_Softmax(output, output);

    return output;
}


f32 NN_Loss(Matrix *prediction, Matrix *target)
{
    if (prediction == NULL || target == NULL)
        return 0.0f;


    f32 loss = 0.0f;

    Matrix out;

    out.rows = 1;
    out.cols = 1;
    out.data = &loss;


    Mat_Cross_Entropy(
        &out,
        prediction,
        target
    );

    return loss;
}

void NN_Train(NeuralNet *nn,
              Matrix *input,
              Matrix *target,
              f32 learning_rate,
              Arena *arena)
{
    if (!nn || !input || !target || !arena)
        return;

    size_t mark = arena->Offset;


    Matrix *hidden = Mat_Create(arena, nn->weights[0]->rows, 1);
    Matrix *output = Mat_Create(arena, nn->weights[1]->rows, 1);

    if (!hidden || !output)
    {
        arena->Offset = mark;
        return;
    }


    // Forward
    Mat_Multiply(hidden, nn->weights[0], input);
    Mat_Add(hidden, hidden, nn->biases[0]);
    Mat_Relu(hidden, hidden);

    Mat_Multiply(output, nn->weights[1], hidden);
    Mat_Add(output, output, nn->biases[1]);
    Mat_Softmax(output, output);



    // Output gradient
    Matrix *d_output =
        Mat_Create(arena, output->rows, output->cols);

    if (!d_output)
    {
        arena->Offset = mark;
        return;
    }


    Mat_Cross_Entropy_Add_Grad(
        d_output,
        output,
        target
    );



    // Hidden gradient BEFORE updating weights
    Matrix *hidden_grad =
        Mat_Create(arena, hidden->rows, 1);

    if (!hidden_grad)
    {
        arena->Offset = mark;
        return;
    }


    for (u32 i = 0; i < hidden->rows; i++)
    {
        hidden_grad->data[i] = 0.0f;

        for (u32 j = 0; j < d_output->rows; j++)
        {
            hidden_grad->data[i] +=
                nn->weights[1]->data[j * nn->weights[1]->cols + i] *
                d_output->data[j];
        }
    }



    Matrix *relu_grad =
        Mat_Create(arena, hidden->rows, 1);

    if (!relu_grad)
    {
        arena->Offset = mark;
        return;
    }


    Mat_Relu_Grad(relu_grad, hidden);

    Mat_Hadamard(
        hidden_grad,
        hidden_grad,
        relu_grad
    );



    // Weight gradients
    Matrix *dW2 =
        Mat_Create(arena,
                   nn->weights[1]->rows,
                   nn->weights[1]->cols);

    Matrix *dW1 =
        Mat_Create(arena,
                   nn->weights[0]->rows,
                   nn->weights[0]->cols);


    if (!dW1 || !dW2)
    {
        arena->Offset = mark;
        return;
    }


    Mat_Outer(
        dW2,
        d_output,
        hidden
    );


    Mat_Outer(
        dW1,
        hidden_grad,
        input
    );



    // Update weights
    Mat_AddScaled(
        nn->weights[1],
        dW2,
        -learning_rate
    );

    Mat_AddScaled(
        nn->biases[1],
        d_output,
        -learning_rate
    );

    Mat_AddScaled(
        nn->weights[0],
        dW1,
        -learning_rate
    );

    Mat_AddScaled(
        nn->biases[0],
        hidden_grad,
        -learning_rate
    );


    // Free temporary matrices
    arena->Offset = mark;
}