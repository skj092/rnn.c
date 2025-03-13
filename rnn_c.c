#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rnn_utils.h"
#include "utils.h"

typedef struct {
    int input_size;
    int hidden_size;
} RNN_Config;

typedef struct {
    RNN_Config config;
    float **weight_ih;  // Array of pointers for each layer's weights
    float **weight_hh;
    float **bias_ih;
    float **bias_hh;
    int num_layers;
} RNN_Model;

void rnn_forward(RNN_Model *model, float *output, float *final_h, float *input,
                int batch_size, int seq_len) {
    int input_size = model->config.input_size;
    int hidden_size = model->config.hidden_size;
    int num_layers = model->num_layers;

    // Allocate hidden states for all layers
    float **h_t = (float **)malloc(num_layers * sizeof(float *));
    for (int l = 0; l < num_layers; l++) {
        h_t[l] = (float *)calloc(batch_size * hidden_size, sizeof(float));
    }

    // Temporary buffers
    float *temp1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp2 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp3 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *x_t = (float *)malloc(batch_size * input_size * sizeof(float));

    // Pre-compute transposed weights for all layers
    float **weight_ih_T = (float **)malloc(num_layers * sizeof(float *));
    float **weight_hh_T = (float **)malloc(num_layers * sizeof(float *));
    for (int l = 0; l < num_layers; l++) {
        weight_ih_T[l] = (float *)malloc((l == 0 ? input_size : hidden_size) * hidden_size * sizeof(float));
        weight_hh_T[l] = (float *)malloc(hidden_size * hidden_size * sizeof(float));
        transpose_matrix(model->weight_ih[l], weight_ih_T[l], hidden_size,
                        l == 0 ? input_size : hidden_size);
        transpose_matrix(model->weight_hh[l], weight_hh_T[l], hidden_size, hidden_size);
    }

    for (int t = 0; t < seq_len; t++) {
        // Extract input for current timestep
        for (int b = 0; b < batch_size; b++) {
            int offset = b * seq_len * input_size + t * input_size;
            memcpy(x_t + b * input_size, input + offset, input_size * sizeof(float));
        }

        // Process each layer
        float *layer_input = x_t;
        for (int layer = 0; layer < num_layers; layer++) {
            // RNN computation
            matrix_multiply(layer_input, weight_ih_T[layer], temp1,
                          batch_size,
                          layer == 0 ? input_size : hidden_size,
                          hidden_size);
            matrix_multiply(h_t[layer], weight_hh_T[layer], temp2,
                          batch_size, hidden_size, hidden_size);
            vector_add(temp1, temp2, temp3, batch_size * hidden_size);

            // Add biases
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < hidden_size; i++) {
                    temp3[b * hidden_size + i] += model->bias_ih[layer][i] +
                                                model->bias_hh[layer][i];
                }
            }

            apply_tanh(temp3, h_t[layer], batch_size * hidden_size);

            // Use current layer's output as next layer's input
            layer_input = h_t[layer];
        }

        // Store output from last layer
        for (int b = 0; b < batch_size; b++) {
            int offset = b * seq_len * hidden_size + t * hidden_size;
            memcpy(output + offset, h_t[num_layers-1] + b * hidden_size,
                  hidden_size * sizeof(float));
        }
    }

    // Copy final hidden states for all layers
    for (int layer = 0; layer < num_layers; layer++) {
        memcpy(final_h + layer * batch_size * hidden_size,
               h_t[layer],
               batch_size * hidden_size * sizeof(float));
    }

    // Cleanup
    for (int l = 0; l < num_layers; l++) {
        free(h_t[l]);
        free(weight_ih_T[l]);
        free(weight_hh_T[l]);
    }
    free(h_t);
    free(weight_ih_T);
    free(weight_hh_T);
    free(temp1);
    free(temp2);
    free(temp3);
    free(x_t);
}

int main() {
    int batch_size = 3;
    int seq_len = 5;
    int input_size = 10;
    int hidden_size = 20;
    int num_layers = 2;  // Example with 2 layers

    RNN_Model model;
    model.config.input_size = input_size;
    model.config.hidden_size = hidden_size;
    model.num_layers = num_layers;

    // Allocate arrays for weights
    model.weight_ih = (float **)malloc(num_layers * sizeof(float *));
    model.weight_hh = (float **)malloc(num_layers * sizeof(float *));
    model.bias_ih = (float **)malloc(num_layers * sizeof(float *));
    model.bias_hh = (float **)malloc(num_layers * sizeof(float *));

    // Load weights for each layer
    int size;
    for (int l = 0; l < num_layers; l++) {
        char filename[50];
        sprintf(filename, "rnn_data/weight_ih_l%d", l);
        model.weight_ih[l] = load_tensor(filename, &size);
        sprintf(filename, "rnn_data/weight_hh_l%d", l);
        model.weight_hh[l] = load_tensor(filename, &size);
        sprintf(filename, "rnn_data/bias_ih_l%d", l);
        model.bias_ih[l] = load_tensor(filename, &size);
        sprintf(filename, "rnn_data/bias_hh_l%d", l);
        model.bias_hh[l] = load_tensor(filename, &size);
    }

    float *input = load_tensor("rnn_data/input_data.npy", &size);
    float *output = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *final_h = (float *)malloc(num_layers * batch_size * hidden_size * sizeof(float));

    rnn_forward(&model, output, final_h, input, batch_size, seq_len);

    float *torch_output = load_tensor("rnn_data/output.bin", &size);
    printf("\nVerifying output:\n");
    check_tensor(torch_output, output, size, "output comparison");

    // Cleanup
    for (int l = 0; l < num_layers; l++) {
        free(model.weight_ih[l]);
        free(model.weight_hh[l]);
        free(model.bias_ih[l]);
        free(model.bias_hh[l]);
    }
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

