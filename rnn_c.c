#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rnn_utils.h"
#include "utils.h"


void rnn_forward(RNN_Cell *model, float *output, float *final_h, float *input,
                 int batch_size, int seq_len) {
    int input_size = model->config.input_size;
    int hidden_size = model->config.hidden_size;

    float *h_t = (float *)calloc(batch_size * hidden_size, sizeof(float));
    float *temp1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp2 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp3 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *x_t = (float *)malloc(batch_size * input_size * sizeof(float));

    float *weight_ih_T = (float *)malloc(input_size * hidden_size * sizeof(float));
    float *weight_hh_T = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    transpose_matrix(model->weight_ih, weight_ih_T, hidden_size, input_size);
    transpose_matrix(model->weight_hh, weight_hh_T, hidden_size, hidden_size);

    for (int t = 0; t < seq_len; t++) {
        // Extract input for current timestep
        for (int b = 0; b < batch_size; b++) {
            int offset = b * seq_len * input_size + t * input_size;
            memcpy(x_t + b * input_size, input + offset, input_size * sizeof(float));
        }

        // RNN computation
        matrix_multiply(x_t, weight_ih_T, temp1, batch_size, input_size, hidden_size);
        matrix_multiply(h_t, weight_hh_T, temp2, batch_size, hidden_size, hidden_size);
        vector_add(temp1, temp2, temp3, batch_size * hidden_size);

        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < hidden_size; i++) {
                temp3[b * hidden_size + i] += model->bias_ih[i] + model->bias_hh[i];
            }
        }

        apply_tanh(temp3, h_t, batch_size * hidden_size);

        // Debug checks (unchanged)
        printf("===== %d =====\n", t);
        if (t == 0) {
            float *act;
            int temp;
            act = load_tensor("./rnn_data/act_0", &temp);
            printf("act length is: %d \n", temp);
            check_tensor(act, h_t, 4, "act0");
            free(act);
        } else {
            float *act;
            int temp;
            act = load_tensor("./rnn_data/act_1", &temp);
            check_tensor(act, h_t, 4, "act1");
            free(act);
        }

        // Store output in batch-first order
        for (int b = 0; b < batch_size; b++) {
            int offset = b * seq_len * hidden_size + t * hidden_size;
            memcpy(output + offset, h_t + b * hidden_size, hidden_size * sizeof(float));
        }
    }

    memcpy(final_h, h_t, batch_size * hidden_size * sizeof(float));

    free(h_t);
    free(temp1);
    free(temp2);
    free(temp3);
    free(x_t);
    free(weight_ih_T);
    free(weight_hh_T);
}

int main() {
    int batch_size = 3;
    int seq_len = 2;
    int input_size = 10;
    int hidden_size = 20;

    RNN_Cell model;
    model.config.input_size = input_size;
    model.config.hidden_size = hidden_size;

    int size;
    model.weight_ih = load_tensor("rnn_data/weight_ih_l0", &size);
    model.weight_hh = load_tensor("rnn_data/weight_hh_l0", &size);
    model.bias_ih = load_tensor("rnn_data/bias_ih_l0", &size);
    model.bias_hh = load_tensor("rnn_data/bias_hh_l0", &size);

    float *input = load_tensor("rnn_data/input_data.npy", &size);
    float *output = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *final_h = (float *)malloc(batch_size * hidden_size * sizeof(float));

    rnn_forward(&model, output, final_h, input, batch_size, seq_len);

    float *torch_output = load_tensor("rnn_data/output.bin", &size);
    printf("\nVerifying output:\n");
    check_tensor(torch_output, output, size, "output comparision");


    free(model.weight_ih);
    free(model.weight_hh);
    free(model.bias_ih);
    free(model.bias_hh);
    free(input);
    free(output);
    free(final_h);
    free(torch_output);

    return 0;
}

