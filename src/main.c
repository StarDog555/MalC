#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/Malc.h"

int main()
{
    Arena arena;
    Arena_Create(&arena, 16 * 1024 * 1024);

    Arena temp;
    Arena_Create(&temp, 16 * 1024 * 1024);


    pcg32_random_t rng;
    PCG32_Seed(&rng, 1234);


    NeuralNet *nn = NN_Create(
        &arena,
        1,      // input
        64,     // hidden nodes
        3,      // classes
        &rng
    );


    if (!nn)
    {
        printf("Network creation failed\n");
        return 1;
    }


    Matrix *input =
        Mat_Create(&arena, 1, 1);

    Matrix *target =
        Mat_Create(&arena, 3, 1);


    if (!input || !target)
    {
        printf("Matrix creation failed\n");
        return 1;
    }



    // Input ranges:
    // 0.0 - 0.9 -> class 0
    // 1.0 - 1.9 -> class 1
    // 2.0 - 2.9 -> class 2

    float training_data[15][2] =
    {
        {0.0f,0},
        {0.2f,0},
        {0.4f,0},
        {0.6f,0},
        {0.8f,0},

        {1.0f,1},
        {1.2f,1},
        {1.4f,1},
        {1.6f,1},
        {1.8f,1},

        {2.0f,2},
        {2.2f,2},
        {2.4f,2},
        {2.6f,2},
        {2.8f,2}
    };


    printf("Training...\n");


    for(int epoch = 0; epoch < 200000; epoch++)
    {
        float loss_sum = 0.0f;


        for(int i = 0; i < 15; i++)
        {
            Arena_Reset(&temp);


            Mat_Clear(target);


            input->data[0] =
                training_data[i][0];


            int label =
                (int)training_data[i][1];


            target->data[label] = 1.0f;



            Matrix *prediction =
                NN_Forward(
                    nn,
                    input,
                    &temp
                );


            if(prediction)
            {
                loss_sum +=
                    NN_Loss(
                        prediction,
                        target
                    );
            }



            NN_Train(
                nn,
                input,
                target,
                0.001f,
                &temp
            );
        }


        if(epoch % 20000 == 0)
        {
            printf(
                "Epoch %d Loss %.5f\n",
                epoch,
                loss_sum
            );
        }
    }


    printf("Training complete\n");



    while(1)
    {
        char buffer[64];


        printf("\nInput number (q to quit): ");


        if(scanf("%63s", buffer) != 1)
            break;


        if(strcmp(buffer,"q") == 0)
            break;



        input->data[0] =
            (float)atof(buffer);



        int expected;


        printf("Expected output (0-2): ");


        if(scanf("%d",&expected) != 1)
            break;



        Mat_Clear(target);


        if(expected >= 0 && expected < 3)
        {
            target->data[expected] = 1.0f;
        }



        Arena_Reset(&temp);


        Matrix *prediction =
            NN_Forward(
                nn,
                input,
                &temp
            );


        if(!prediction)
        {
            printf("Prediction failed\n");
            continue;
        }



        printf("Prediction:\n");


        for(u32 i = 0;
            i < prediction->rows;
            i++)
        {
            printf(
                "%.3f\n",
                prediction->data[i]
            );
        }



        printf(
            "Loss: %.5f\n",
            NN_Loss(
                prediction,
                target
            )
        );
    }



    Arena_Destroy(&temp);
    Arena_Destroy(&arena);

    return 0;
}