#include <float.h>
#include <math.h>
#include <stdio.h>

#include "rnn_utils.h"
#include "utils.h"

// Linear projection: Y = X * W^T + b
void linear_forward(float *X, float *W, float *b, float *Y, int batch_size,
                    int seq_len, int input_dim, int output_dim) {
  for (int bs = 0; bs < batch_size; bs++) {
    for (int seq = 0; seq < seq_len; seq++) {
      float *x_ptr = X + (bs * seq_len + seq) * input_dim;
      float *y_ptr = Y + (bs * seq_len + seq) * output_dim;

      for (int i = 0; i < output_dim; i++) {
        y_ptr[i] = b[i]; // Add bias
        for (int j = 0; j < input_dim; j++) {
          y_ptr[i] += x_ptr[j] * W[i * input_dim + j];
        }
      }
    }
  }
}

void log_softmax_forward_cpu(float *out, const float *inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    // Find max value for numerical stability
    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }

    // Compute logsumexp
    double sum = 0.0;
    for (int j = 0; j < C; j++) {
      sum += expf(inp_row[j] - maxval);
    }
    float logsumexp = logf(sum) + maxval;

    // Compute log softmax
    for (int j = 0; j < C; j++) {
      out_row[j] = inp_row[j] - logsumexp;
    }
  }
}
typedef struct {
  float *weight;
  float *bias;
} linear;

typedef struct {
  RNN_Model rnn;
  linear h2o;
} CharRNN;

void initialize_model(CharRNN *model, int n_input, int n_hidden, int n_output) {
  int num_layers = 1;
  model->rnn.config.input_size = n_input;
  model->rnn.config.hidden_size = n_hidden;
  model->rnn.num_layers = num_layers;

  // Allocate arrays for weights
  model->rnn.weight_ih = (float **)malloc(num_layers * sizeof(float *));
  model->rnn.weight_hh = (float **)malloc(num_layers * sizeof(float *));
  model->rnn.bias_ih = (float **)malloc(num_layers * sizeof(float *));
  model->rnn.bias_hh = (float **)malloc(num_layers * sizeof(float *));

  // Load weights for each layer
  int size;
  char filename[50];
  int l = 0;
  sprintf(filename, "modeltest/weight_ih_l%d", l);
  model->rnn.weight_ih[l] = load_tensor(filename, &size);
  sprintf(filename, "modeltest/weight_hh_l%d", l);
  model->rnn.weight_hh[l] = load_tensor(filename, &size);
  sprintf(filename, "modeltest/bias_ih_l%d", l);
  model->rnn.bias_ih[l] = load_tensor(filename, &size);
  sprintf(filename, "modeltest/bias_hh_l%d", l);
  model->rnn.bias_hh[l] = load_tensor(filename, &size);

  // load weight of linear layer
  sprintf(filename, "modeltest/lw");
  model->h2o.weight = load_tensor(filename, &size); // (10, 128)
  sprintf(filename, "modeltest/lb");
  model->h2o.bias = load_tensor(filename, &size); // (10)
}

void forward_CharRNN(CharRNN *model, float *output,
                     float *input) {

  // run rnn forward
  float *final_h = (float *)malloc(128 * sizeof(float));
  float *rnn_out = (float *)malloc(1 * 9 * 128 * sizeof(float));
  rnn_forward(&model->rnn, rnn_out, final_h, input, 1, 9);

  // verify with pytorch output
  int size;
  float *torch_output = load_tensor("./modeltest/rnn_out", &size);
  float *torch_hidden = load_tensor("./modeltest/hidden", &size);
  check_tensor(rnn_out, torch_output, size, "out");
  check_tensor(final_h, torch_hidden, size, "hidden");

  // run linear layer
  float *lo = (float *)malloc(9 * 1 * 10 * sizeof(float));
  linear_forward(final_h, model->h2o.weight, model->h2o.bias, lo, 9, 1, 128,
                 10);
  // verify with pytorch output
  float *lo_tensor = load_tensor("modeltest/lo", &size);
  check_tensor(lo_tensor, lo, size, "linear output ->");

  // run softmax
  log_softmax_forward_cpu(output, lo, 1, 10);
  float *so_pt = load_tensor("modeltest/out.bin", &size);

  //verify
  check_tensor(so_pt, output, size, "fnal output");

  free(lo);
  free(final_h);
  free(torch_output);
  free(torch_hidden);
  free(lo_tensor);
}
void free_model(CharRNN *model) {
  int num_layers = model->rnn.num_layers;

  for (int l = 0; l < num_layers; l++) {
    free(model->rnn.weight_ih[l]);
    free(model->rnn.weight_hh[l]);
    free(model->rnn.bias_ih[l]);
    free(model->rnn.bias_hh[l]);
  }

  free(model->rnn.weight_ih);
  free(model->rnn.weight_hh);
  free(model->rnn.bias_ih);
  free(model->rnn.bias_hh);

  free(model->h2o.weight);
  free(model->h2o.bias);
}

int main() {
  int n_input = 26;
  int n_hidden = 128;
  int n_output = 10;
  int num_layers = 1;

  CharRNN model;
  initialize_model(&model, n_input, n_hidden, n_output);

  int size;
  float *input = load_tensor("./modeltest/x.bin", &size); // (9, 1, 26)
  float *output = (float *)malloc(1 * n_output * sizeof(float));

  forward_CharRNN(&model, output, input);

  free(input);
  free(output);
  free_model(&model);

  return 0;
}
