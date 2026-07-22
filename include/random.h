#pragma once

#include <stdint.h>
#include "Types.h"
#include "Matrix.h"

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;


// Main functions

uint32_t pcg32_random_r(pcg32_random_t* rng);
uint32_t random_range(pcg32_random_t* rng, uint32_t min, uint32_t max);

// Helper Functions

void PCG32_Seed(pcg32_random_t *rng, uint64_t seed);
float random_float(pcg32_random_t *rng, float min, float max);
void Mat_Random(Matrix *m, pcg32_random_t *rng, f32 min, f32 max);

void Mat_Xavier_Init(Matrix *m, pcg32_random_t *rng, u32 fan_in, u32 fan_out);
void Mat_He_Init(Matrix *m, pcg32_random_t *rng, u32 fan_in);
