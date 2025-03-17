#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load tensor without shape file
float *load_tensor(const char *bin_path, int *total_size) {
  FILE *bin_file = fopen(bin_path, "rb");
  if (!bin_file) {
    fprintf(stderr, "Error: Could not open file %s\n", bin_path);
    return NULL;
  }

  // Get file size
  fseek(bin_file, 0, SEEK_END);
  long file_size = ftell(bin_file);
  fseek(bin_file, 0, SEEK_SET);

  *total_size = file_size / sizeof(float);

  // Allocate memory and read data
  float *data = (float *)malloc(file_size);
  if (!data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(bin_file);
    return NULL;
  }

  size_t read_count = fread(data, 1, file_size, bin_file);
  if (read_count != file_size) {
    fprintf(stderr, "Warning: Expected to read %ld bytes, but got %ld\n",
            file_size, read_count);
  }

  fclose(bin_file);
  return data;
}

// Matrix multiplication: C = A * B -> (m, k) x (k, n) -> (m, n)
void matrix_multiply(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

// Vector addition: C = A + B
void vector_add(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Apply tanh activation
void apply_tanh(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        B[i] = tanhf(A[i]);
    }
}

// RNN configuration struct
typedef struct {
    int input_size;
    int hidden_size;
    int num_layers;
    bool batch_first;
} RNN_Conf;

// Default configuration
RNN_Conf default_config() {
    return (RNN_Conf){10, 20, 1, true};
}

// RNN model struct
typedef struct {
    float *weight_ih;
    float *weight_hh;
    float *bias_ih;
    float *bias_hh;
    float *weight_ih_t;
    float *weight_hh_t;
    RNN_Conf config;
} RNN_Cell;

// Load pretrained weights
void load_pretrained_weight(RNN_Cell *rnn) {
    int total_size;
    rnn->weight_ih = load_tensor("rnn_data/weight_ih_l0", &total_size);
    rnn->weight_hh = load_tensor("rnn_data/weight_hh_l0", &total_size);
    rnn->bias_ih = load_tensor("rnn_data/bias_ih_l0", &total_size);
    rnn->bias_hh = load_tensor("rnn_data/bias_hh_l0", &total_size);
}

void transpose_matrix(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}


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

