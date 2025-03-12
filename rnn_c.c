#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "rnn_utils.h"


void rnn_forward(RNN_Cell *model, float *output, float *final_h, float *input,
                 int batch_size, int seq_len) {
  int input_size = model->config.input_size;
  int hidden_size = model->config.hidden_size;

  float *h_t = (float *)calloc(batch_size * hidden_size, sizeof(float));
  float *temp1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
  float *temp2 = (float *)malloc(batch_size * hidden_size * sizeof(float));
  float *temp3 = (float *)malloc(batch_size * hidden_size * sizeof(float));
  float *x_t = (float *)malloc(batch_size * input_size * sizeof(float));

  float *weight_ih_T =
      (float *)malloc(input_size * hidden_size * sizeof(float));
  float *weight_hh_T =
      (float *)malloc(hidden_size * hidden_size * sizeof(float));

  transpose_matrix(model->weight_ih, weight_ih_T, hidden_size, input_size);
  transpose_matrix(model->weight_hh, weight_hh_T, hidden_size, hidden_size);

  memcpy(x_t, input, batch_size * input_size * sizeof(float));

  matrix_multiply(x_t, weight_ih_T, temp1, batch_size, input_size, hidden_size);
  matrix_multiply(h_t, weight_hh_T, temp2, batch_size, hidden_size,
                  hidden_size);
  vector_add(temp1, temp2, temp3, batch_size * hidden_size);

  // Add biases correctly for each batch
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < hidden_size; i++) {
      temp3[b * hidden_size + i] += model->bias_ih[i] + model->bias_hh[i];
    }
  }

  apply_tanh(temp3, h_t, batch_size * hidden_size);

  memcpy(output, h_t, batch_size * hidden_size * sizeof(float));
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
    RNN_Conf config = default_config();
    RNN_Cell rnn = {0};
    rnn.config = config;
    load_pretrained_weight(&rnn);

    int batch_size = 3, seq_len = 1;
    int input_size = config.input_size; // 10
    int hidden_size = config.hidden_size; // 20

    float *input = (float *)malloc(batch_size * seq_len * input_size * sizeof(float));
    float *output = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *expected = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *final_h = (float *)malloc(batch_size * hidden_size * sizeof(float));

    // load pytorch input for verification
    int total_size;
    input = load_tensor("./rnn_data/input_data.npy", &total_size);
    // print_float_array(input, batch_size * seq_len * input_size);
    expected = load_tensor("./rnn_data/output.bin", &total_size);


    output = (float *)malloc(3 * 5 * 10 * sizeof(float));
    rnn_forward(&rnn, output, final_h, input, batch_size, seq_len);
    // print_float_array(output, 40);
    check_tensor(output, expected, batch_size * seq_len * hidden_size, "output");

    free(input);
    free(expected);
    free(output);
    free(final_h);
    free(rnn.weight_ih);
    free(rnn.weight_hh);
    free(rnn.bias_ih);
    free(rnn.bias_hh);

    return 0;
}

