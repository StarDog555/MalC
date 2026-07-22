#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "../include/random.h"

// 32-bit random integer

uint32_t pcg32_random_r(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Bounded range function [min, max] using Lemire's unbiased divisionless method

uint32_t random_range(pcg32_random_t* rng, uint32_t min, uint32_t max) {
    uint32_t range = max - min + 1;
    uint64_t random_32bit = pcg32_random_r(rng);
    
    // Fast multiplication bounding
    uint64_t multi_result = random_32bit * range;
    uint32_t low_bits = (uint32_t)multi_result;
    
    // Reject biased samples if the range does not evenly divide 2^32
    if (low_bits < range) {
        uint32_t threshold = -range % range;
        while (low_bits < threshold) {
            random_32bit = pcg32_random_r(rng);
            multi_result = random_32bit * range;
            low_bits = (uint32_t)multi_result;
        }
    }
    return min + (multi_result >> 32);
}

void PCG32_Seed(pcg32_random_t *rng, uint64_t seed)
{
    rng->state = 0;
    rng->inc = (seed << 1u) | 1u;

    pcg32_random_r(rng);
    rng->state += seed;
    pcg32_random_r(rng);
}

float random_float(pcg32_random_t *rng, float min, float max)
{
    uint32_t value = pcg32_random_r(rng);
    float normalized = (float)value / (float)UINT32_MAX;
    return min + normalized * (max - min);
}


void Mat_Xavier_Init(Matrix *m, pcg32_random_t *rng, u32 fan_in, u32 fan_out)
{
    if (m == NULL || m->data == NULL)
        return;

    f32 limit = sqrtf(6.0f / (fan_in + fan_out));

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = random_float(rng, -limit, limit);
    }
}

void Mat_He_Init(Matrix *m, pcg32_random_t *rng, u32 fan_in)
{
    if (m == NULL || m->data == NULL)
        return;

    f32 std = sqrtf(2.0f / fan_in);

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        f32 r = random_float(rng, -1.0f, 1.0f);

        m->data[i] = r * std;
    }
}

void Mat_Random(Matrix *m, pcg32_random_t *rng, f32 min, f32 max)
{
    if (m == NULL || m->data == NULL)
        return;

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] = random_float(rng, min, max);
    }
}