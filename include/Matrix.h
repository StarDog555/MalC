#pragma once

#include "arena.h"
#include "Types.h"

typedef struct Matrix
{
    u32 rows;
    u32 cols;
    f32 *data;
} Matrix;

// ----------------
// Matrix Functions
// ----------------

// Matrix Helper Functions

Matrix *Mat_Create(Arena *a, u32 rows, u32 cols);
void Mat_Clear(Matrix *m);
void Mat_Copy(Matrix *dst, Matrix *src);
void Mat_Fill(Matrix *m, f32 data);
void Mat_Scale(Matrix *m, f32 scale);
void Mat_Print(Matrix *m);

b32 Mat_Transpose(Matrix *out, Matrix *in);
f32 Mat_Dot(Matrix *a, Matrix *b);

// Matrix Math

b32 Mat_Add(Matrix *out, const Matrix *a, const Matrix *b);
b32 Mat_Subtract(Matrix *out, const Matrix *a, const Matrix *b);
b32 Mat_Multiply(Matrix *out, const Matrix *a, const Matrix *b);
b32 Mat_Relu(Matrix *out, const Matrix *in);
b32 Mat_Softmax(Matrix *out, const Matrix *in);

b32 Mat_Cross_Entropy(Matrix *out, 
const Matrix *p, const Matrix *q);

b32 Mat_Relu_Grad(Matrix *out, Matrix *in);
b32 Mat_Softmax_Add_Grad(Matrix *out, Matrix *softmax_out);

b32 Mat_Cross_Entropy_Add_Grad(Matrix *out, 
const Matrix *p, const Matrix *q);

b32 Mat_Hadamard(Matrix *out, const Matrix *a, const Matrix *b);
b32 Mat_Outer(Matrix *out, const Matrix *a, const Matrix *b);
void Mat_AddScaled(Matrix *dst, const Matrix *grad, f32 scale);
