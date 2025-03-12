#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int input_size;
    int hidden_size;
} RNNConfig;

typedef struct {
    RNNConfig config;
    float *weight_ih;  // [hidden_size, input_size]
    float *weight_hh;  // [hidden_size, hidden_size]
    float *bias_ih;    // [hidden_size]
    float *bias_hh;    // [hidden_size]
} RNN_Cell;

float* load_tensor(const char* filename, int* size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    *size = file_size / sizeof(float);
    float *data = (float *)malloc(file_size);
    fread(data, sizeof(float), *size, fp);
    fclose(fp);
    return data;
}

void print_float_array(const float* arr, int size, const char* label) {
    printf("%s: ", label);
    for (int i = 0; i < size; i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

void transpose_matrix(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

void vector_add(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void apply_tanh(const float* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = tanhf(src[i]);
    }
}

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
    int batch_size = 3;
    int seq_len = 1;
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
    float *output = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *final_h = (float *)malloc(batch_size * hidden_size * sizeof(float));

    rnn_forward(&model, output, final_h, input, batch_size, seq_len);

    float *torch_output = load_tensor("rnn_data/output.bin", &size);
    float *torch_hn = load_tensor("rnn_data/hn.bin", &size);

    printf("\nVerifying output:\n");
    for (int i = 0; i < batch_size * hidden_size; i++) {
        float diff = fabs(output[i] - torch_output[i]);
        printf("%s %.6f %.6f (diff: %.6f)\n",
               (diff < 1e-4) ? "OK" : "NOT OK",
               output[i], torch_output[i], diff);
    }

    free(model.weight_ih);
    free(model.weight_hh);
    free(model.bias_ih);
    free(model.bias_hh);
    free(input);
    free(output);
    free(final_h);
    free(torch_output);
    free(torch_hn);

    return 0;
}
