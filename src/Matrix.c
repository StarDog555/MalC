#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/Matrix.h"
#include "../include/Types.h"
#include "../include/random.h"

// ----------------
// Helper Functions
// ----------------

static int CheckCreate(Arena *a, u32 rows, u32 cols)
{
    if (a == NULL || rows == 0 || cols == 0)
    {
        printf("[-] Error: Mat_Create Failed, Invalid Value\n");
        return 1;
    }

    return 0;
}


// ----------------
// Matrix Creation
// ----------------

Matrix *Mat_Create(Arena *a, u32 rows, u32 cols)
{
    if (CheckCreate(a, rows, cols))
        return NULL;


    Matrix *out = Arena_Alloc(a, sizeof(Matrix));

    if (out == NULL)
        return NULL;


    out->rows = rows;
    out->cols = cols;


    out->data = Arena_Alloc(a, rows * cols * sizeof(f32));

    if (out->data == NULL)
        return NULL;


    return out;
}


// ----------------
// Matrix Helpers
// ----------------

void Mat_Clear(Matrix *m)
{
    if (m == NULL || m->data == NULL)
        return;


    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = 0.0f;
    }
}


void Mat_Copy(Matrix *dst, const Matrix *src)
{
    if (dst == NULL || src == NULL)
        return;


    if (dst->rows != src->rows ||
        dst->cols != src->cols)
    {
        printf("[-] Error: Matrix Size Mismatch\n");
        return;
    }


    for (u32 i = 0; i < src->rows * src->cols; i++)
    {
        dst->data[i] = src->data[i];
    }
}


void Mat_Fill(Matrix *m, f32 value)
{
    if (m == NULL || m->data == NULL)
        return;


    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = value;
    }
}


void Mat_Scale(Matrix *m, f32 scale)
{
    if (m == NULL || m->data == NULL)
        return;


    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] *= scale;
    }
}


// ----------------
// Matrix Math
// ----------------

b32 Mat_Add(Matrix *out,
            const Matrix *a,
            const Matrix *b)
{
    if (out == NULL || a == NULL || b == NULL)
        return false;


    if (a->rows != b->rows ||
        a->cols != b->cols)
        return false;


    for (u32 i = 0; i < a->rows * a->cols; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }


    return true;
}


b32 Mat_Subtract(Matrix *out,
                 const Matrix *a,
                 const Matrix *b)
{
    if (out == NULL || a == NULL || b == NULL)
        return false;


    if (a->rows != b->rows ||
        a->cols != b->cols)
        return false;


    for (u32 i = 0; i < a->rows * a->cols; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }


    return true;
}


b32 Mat_Multiply(Matrix *out,
                 const Matrix *a,
                 const Matrix *b)
{
    if (out == NULL || a == NULL || b == NULL)
        return false;


    if (a->cols != b->rows)
        return false;


    if (out->rows != a->rows ||
        out->cols != b->cols)
        return false;


    for (u32 i = 0; i < a->rows; i++)
    {
        for (u32 j = 0; j < b->cols; j++)
        {
            f32 sum = 0.0f;


            for (u32 k = 0; k < a->cols; k++)
            {
                sum += a->data[i * a->cols + k] *
                       b->data[k * b->cols + j];
            }


            out->data[i * out->cols + j] = sum;
        }
    }


    return true;
}


// ----------------
// Activations
// ----------------

b32 Mat_Relu(Matrix *out,
             const Matrix *in)
{
    if (out == NULL || in == NULL)
        return false;


    if (out->rows != in->rows ||
        out->cols != in->cols)
        return false;


    u64 size = (u64)out->rows * out->cols;


    for (u64 i = 0; i < size; i++)
    {
        out->data[i] =
            in->data[i] > 0.0f ?
            in->data[i] :
            0.0f;
    }


    return true;
}


b32 Mat_Softmax(Matrix *out,
                const Matrix *in)
{
    if (out == NULL || in == NULL)
        return false;


    if (out->rows != in->rows ||
        out->cols != in->cols)
        return false;


    f32 max = in->data[0];


    for (u32 i = 1; i < in->rows * in->cols; i++)
    {
        if (in->data[i] > max)
            max = in->data[i];
    }


    f32 sum = 0.0f;


    for (u32 i = 0; i < in->rows * in->cols; i++)
    {
        out->data[i] = expf(in->data[i] - max);
        sum += out->data[i];
    }


    for (u32 i = 0; i < out->rows * out->cols; i++)
    {
        out->data[i] /= sum;
    }


    return true;
}


// ----------------
// Loss Functions
// ----------------

b32 Mat_Cross_Entropy(Matrix *out,
                      const Matrix *p,
                      const Matrix *q)
{
    if (out == NULL || p == NULL || q == NULL)
    {
        printf("[-] Error: Invalid Matrix\n");
        return false;
    }


    if (p->rows != q->rows ||
        p->cols != q->cols)
    {
        printf("[-] Error: Matrix Size Mismatch\n");
        return false;
    }


    f32 loss = 0.0f;


    for (u32 i = 0; i < p->rows * p->cols; i++)
    {
        f32 prediction = p->data[i];


        if (prediction < 1e-7f)
            prediction = 1e-7f;

        loss += q->data[i] * logf(prediction);
    }

    loss = -loss;

    out->rows = 1;
    out->cols = 1;
    out->data[0] = loss;

    return true;
}

// ----------------
// Gradient Functions
// ----------------

b32 Mat_Relu_Grad(Matrix *out, Matrix *in)
{
    if (out == NULL || in == NULL)
    {
        printf("[-] Error: Mat_Relu_Grad Failed, Invalid Matrix\n");
        return false;
    }

    if (out->rows != in->rows ||
        out->cols != in->cols)
    {
        printf("[-] Error: Mat_Relu_Grad Failed, Size Mismatch\n");
        return false;
    }


    for (u32 i = 0; i < in->rows * in->cols; i++)
    {
        if (in->data[i] > 0.0f)
            out->data[i] = 1.0f;
        else
            out->data[i] = 0.0f;
    }


    return true;
}


b32 Mat_Softmax_Add_Grad(Matrix *out,
                         Matrix *softmax_out)
{
    if (out == NULL || softmax_out == NULL)
    {
        printf("[-] Error: Mat_Softmax_Add_Grad Failed, Invalid Matrix\n");
        return false;
    }


    if (out->rows != softmax_out->rows ||
        out->cols != softmax_out->cols)
    {
        printf("[-] Error: Mat_Softmax_Add_Grad Failed, Size Mismatch\n");
        return false;
    }

    for (u32 i = 0; 
         i < softmax_out->rows * softmax_out->cols; 
         i++)
    {
        f32 s = softmax_out->data[i];

        out->data[i] = s * (1.0f - s);
    }


    return true;
}


b32 Mat_Cross_Entropy_Add_Grad(Matrix *out,
                               const Matrix *p,
                               const Matrix *q)
{
    if (out == NULL || p == NULL || q == NULL)
    {
        printf("[-] Error: Mat_Cross_Entropy_Add_Grad Failed, Invalid Matrix\n");
        return false;
    }


    if (p->rows != q->rows ||
        p->cols != q->cols ||
        out->rows != p->rows ||
        out->cols != p->cols)
    {
        printf("[-] Error: Mat_Cross_Entropy_Add_Grad Failed, Size Mismatch\n");
        return false;
    }


    /*
       Gradient of:
       
       CrossEntropy(Softmax(x), target)

       = prediction - target

    */

    for (u32 i = 0; 
         i < p->rows * p->cols; 
         i++)
    {
        out->data[i] = p->data[i] - q->data[i];
    }


    return true;
}

void Mat_Print(Matrix *m)
{
    if (m == NULL || m->data == NULL)
        return;

    for (u32 i = 0; i < m->rows; i++)
    {
        for (u32 j = 0; j < m->cols; j++)
        {
            printf("%.3f ", 
                m->data[i * m->cols + j]);
        }

        printf("\n");
    }
}


b32 Mat_Transpose(Matrix *out, Matrix *in)
{
    if (out == NULL || in == NULL)
        return false;


    if (out->rows != in->cols ||
        out->cols != in->rows)
    {
        printf("[-] Error: Transpose Size Mismatch\n");
        return false;
    }


    for (u32 i = 0; i < in->rows; i++)
    {
        for (u32 j = 0; j < in->cols; j++)
        {
            out->data[j * out->cols + i] =
                in->data[i * in->cols + j];
        }
    }


    return true;
}

f32 Mat_Dot(Matrix *a, Matrix *b)
{
    if (a == NULL || b == NULL)
        return 0.0f;

    if (a->rows * a->cols !=
        b->rows * b->cols)
    {
        printf("[-] Error: Dot Size Mismatch\n");
        return 0.0f;
    }

    f32 result = 0.0f;

    for (u32 i = 0; i < a->rows * a->cols; i++)
    {
        result += a->data[i] * b->data[i];
    }

    return result;
}

// ----------------
// Extra Matrix Operations
// ----------------

b32 Mat_Hadamard(Matrix *out,
                 const Matrix *a,
                 const Matrix *b)
{
    if (out == NULL || a == NULL || b == NULL)
        return false;

    if (a->rows != b->rows ||
        a->cols != b->cols)
        return false;

    if (out->rows != a->rows ||
        out->cols != a->cols)
        return false;

    u32 size = a->rows * a->cols;

    for (u32 i = 0; i < size; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }

    return true;
}


// ----------------
// Outer Product
// ----------------

b32 Mat_Outer(Matrix *out,
              const Matrix *a,
              const Matrix *b)
{
    if (out == NULL || a == NULL || b == NULL)
        return false;

    if (a->cols != 1 ||
        b->cols != 1)
        return false;

    if (out->rows != a->rows ||
        out->cols != b->rows)
        return false;

    for (u32 i = 0; i < a->rows; i++)
    {
        for (u32 j = 0; j < b->rows; j++)
        {
            out->data[i * out->cols + j] =
                a->data[i] * b->data[j];
        }
    }

    return true;
}


// ----------------
// Add Scaled Gradient
// ----------------

void Mat_AddScaled(Matrix *dst,
                   const Matrix *grad,
                   f32 scale)
{
    if (dst == NULL || grad == NULL)
        return;

    if (dst->rows != grad->rows ||
        dst->cols != grad->cols)
        return;

    u32 size = dst->rows * dst->cols;

    for (u32 i = 0; i < size; i++)
    {
        dst->data[i] += grad->data[i] * scale;
    }
}