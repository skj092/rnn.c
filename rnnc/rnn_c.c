#include <stdio.h>
#include <stdlib.h>
#include "rnn_utils.h"
#include "utils.h"


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

