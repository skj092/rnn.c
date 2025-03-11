#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load tensor without shape file
// Load tensor without shape file (we'll provide the shape directly)
float* load_tensor(const char* bin_path, int* total_size) {
    FILE* bin_file = fopen(bin_path, "rb");
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
    float* data = (float*)malloc(file_size);
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(bin_file);
        return NULL;
    }

    size_t read_count = fread(data, 1, file_size, bin_file);
    if (read_count != file_size) {
        fprintf(stderr, "Warning: Expected to read %ld bytes, but got %ld\n", file_size, read_count);
    }

    fclose(bin_file);
    return data;
}

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char *label) {
  int print_upto = 4;
  int ok = 1;
  float maxdiff = 0.0f;
  float tol = 2e-2f;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    // look at the diffence at position i of these two tensors
    float diff = fabsf(a[i] - b[i]);

    // keep track of the overall error
    ok = ok && (diff <= tol);
    if (diff > maxdiff) {
      maxdiff = diff;
    }

    // for the first few elements of each tensor, pretty print
    // the actual numbers, so we can do a visual, qualitative proof/assessment
    if (i < print_upto) {
      if (diff <= tol) {
        if (i < print_upto) {
          printf("OK ");
        }
      } else {
        if (i < print_upto) {
          printf("NOT OK ");
        }
      }
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result for this tensor
  if (ok) {
    printf("TENSOR OK, maxdiff = %e\n", maxdiff);
  } else {
    printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
  }
  return ok;
}


// Helper function to print an array of floats
void print_float_array(float *array, size_t size) {
  for (size_t i = 0; i < size; i++) {
    printf("%.6f ", array[i]);
  }
  printf("\n");
}


// Matrix multiplication: C = A * B
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
} RNN_Model;

// Load pretrained weights
void load_pretrained_weight(RNN_Model *rnn) {
    int total_size;
    rnn->weight_ih = load_tensor("rnn_data/weight_ih_l0", &total_size);
    rnn->weight_hh = load_tensor("rnn_data/weight_hh_l0", &total_size);
    rnn->bias_ih = load_tensor("rnn_data/bias_ih_l0", &total_size);
    rnn->bias_hh = load_tensor("rnn_data/bias_hh_l0", &total_size);
    // print_float_array(rnn->weight_ih, 5);
    // print_float_array(rnn->weight_hh, 5);
    // print_float_array(rnn->bias_ih, 5);
    // print_float_array(rnn->bias_hh, 5);
}
void transpose_matrix(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}


void rnn_forward(RNN_Model *model, float *output, float *final_h, float *input, int batch_size, int seq_len) {
    int input_size = model->config.input_size;
    int hidden_size = model->config.hidden_size;

    float *h_t = (float *)calloc(batch_size * hidden_size, sizeof(float));
    float *temp1 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp2 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *temp3 = (float *)malloc(batch_size * hidden_size * sizeof(float));
    float *x_t = (float *)malloc(batch_size * input_size * sizeof(float));

    // Allocate space for transposed weights
    float *weight_ih_T = (float *)malloc(input_size * hidden_size * sizeof(float));
    float *weight_hh_T = (float *)malloc(hidden_size * hidden_size * sizeof(float));

    // Transpose weights
    transpose_matrix(model->weight_ih, weight_ih_T, hidden_size, input_size);
    transpose_matrix(model->weight_hh, weight_hh_T, hidden_size, hidden_size);

    for (int t = 0; t < 1; t++) {
        // Extract x[:, t, :]
        for (int b = 0; b < batch_size; b++) {
            memcpy(&x_t[b * input_size], &input[b * (seq_len * input_size) + t * input_size], input_size * sizeof(float));
        }

        // print_float_array(x_t, batch_size * input_size);  // Debug output

        // Matrix multiplications using transposed weights
        matrix_multiply(x_t, weight_ih_T, temp1, batch_size, input_size, hidden_size);
        matrix_multiply(h_t, weight_hh_T, temp2, batch_size, hidden_size, hidden_size);

        // Compute activations
        vector_add(temp1, temp2, temp3, batch_size * hidden_size);
        vector_add(temp3, model->bias_ih, temp3, batch_size * hidden_size);
        vector_add(temp3, model->bias_hh, temp3, batch_size * hidden_size);
        apply_tanh(temp3, h_t, batch_size * hidden_size);

        // Store the output
        memcpy(&output[t * batch_size * hidden_size], h_t, batch_size * hidden_size * sizeof(float));
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
    RNN_Conf config = default_config();
    RNN_Model rnn = {0};
    rnn.config = config;
    load_pretrained_weight(&rnn);

    int batch_size = 3, seq_len = 5;
    int input_size = config.input_size;
    int hidden_size = config.hidden_size;

    float *input = (float *)malloc(batch_size * seq_len * input_size * sizeof(float));
    float *output = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *expected = (float *)malloc(batch_size * seq_len * hidden_size * sizeof(float));
    float *final_h = (float *)malloc(batch_size * hidden_size * sizeof(float));

    // load pytorch input for verification
    int total_size;
    input = load_tensor("./rnn_data/input_data.npy", &total_size);
    // print_float_array(input, 152);
    expected = load_tensor("./rnn_data/output.bin", &total_size);


    output = (float *)malloc(3 * 5 * 10 * sizeof(float));
    rnn_forward(&rnn, output, final_h, input, batch_size, seq_len);
    check_tensor(output, expected, 150, "output");

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

