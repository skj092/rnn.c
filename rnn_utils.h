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

