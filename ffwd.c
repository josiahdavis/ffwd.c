#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct {
    float *W, *b;
    int input_size, output_size;
} Linear;

typedef struct {
    Linear *layer; // Define FFWD as an array of linear layers
    int num_layers;
} FeedForward;

void matmul_bias(float* x, float* Wx, float* bias, float* out, int B, int C_in, int C_out){
    /*
    Computes: 
        out  =  x @ Wx + bias
    Where:
         x    (B,    C_in)
         Wx   (C_in, C_out)
         bias (C_out,     )
         out  (B,    C_out)
    */

    for (int b = 0; b < B; b++){
        for (int c = 0; c < C_out; c++){
            // compute dot product of row of x and column of Wx.
            float dot = (bias != NULL) ? bias[c] : 0.0f;
            for (int i = 0; i < C_in; i++){
                // formula for indexing into a matrix to get correct element:
                //      row index * column width + column index
                dot += x[b * C_in + i] * Wx[i * C_out + c];
            }
            out[b * C_out + c] = dot;
        }
    }
}

void relu(float *v, int size){
    for (int i = 0; i < size; i++){
        if (v[i] < 0) v[i] = 0;
    }
}

void forward(float *x, int B, int C_in, Linear *layer_in, Linear *layer1, Linear *layer2
    , Linear *layer3, Linear *layer_out, float* out){
    // Input Projection, followed by Relu
    float* act1 = (float*) malloc(B * layer_in->output_size * sizeof(float));
    matmul_bias(x, layer_in->W, layer_in->b, act1, B, C_in, layer_in->output_size);
    relu(act1, B * layer_in->output_size);

    // Linear, followed by Relu
    float* act2 = (float*) malloc(B * layer1->output_size * sizeof(float));
    // matmul_bias(act1, W1, b1, act2, B, C, C);
    matmul_bias(act1, layer1->W, layer1->b, act2, B, layer1->input_size, layer1->output_size);
    relu(act2, B * layer1->output_size);
    
    // Linear, followed by Relu
    float* act3 = (float*) malloc(B * layer2->output_size * sizeof(float));
    // matmul_bias(act2, W2, b2, act3, B, C, C);
    matmul_bias(act2, layer2->W, layer2->b, act3, B, layer2->input_size, layer2->output_size);
    relu(act3, B * layer2->output_size);
    
    // Linear, followed by Relu
    float* act4 = (float*) malloc(B * layer3->output_size * sizeof(float));
    // matmul_bias(act3, W3, b3, act4, B, C, C);
    matmul_bias(act3, layer3->W, layer3->b, act4, B, layer3->input_size, layer3->output_size);
    relu(act4, B * layer3->output_size);
    
    // Output projection
    // matmul_bias(act4, W_out, b_out, out, B, C, 1);
    matmul_bias(act4, layer_out->W, layer_out->b, out, B, layer_out->input_size, layer_out->output_size);
    free(act1);
    free(act2);
    free(act3);
    free(act4);
}

void read_csv(float* data, const char* file_location, int nrows, int ncols) {
    // Poor man's csv file loader.
    // Assumes all data is numerical with no header. 
    // Stores in row-wise format.
    
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    FILE *file = fopen(file_location, "r");
    if (file == NULL){
        fprintf(stderr, "Failed to open file: %s\n", file_location);
        return;
    }    

    char *line = NULL;
    size_t len = 0; // size_t?
    ssize_t read; // ssize_t?
    
    // Iterate through rows of the CSV file
    for (int i = 0; i < nrows; i++){
        // Read single row of data
        // getline dynamically allocates memory. fgets requires pre-allocated buffer.
        read = getline(&line, &len, file); 
        if (read == -1) {
            fprintf(stderr, "Error: not enough rows in CSV file (expected %d)\n", nrows);
            fclose(file);
            free(line);
            return;
        }
        char *token;
        char *saveptr;
        
        // Iterate through colunns of the individual row
        // strtok_r preferred to strtok for thread-safety
        token = strtok_r(line, ",", &saveptr); // gets first character in the row
        for (int j = 0; j < ncols; j++){
            if (token == NULL){
                fprintf(stderr, "Error: Not enough columns in row %d (expected %d)\n", i+1, ncols);
                fclose(file);
                free(line);
                return;
            }
            // Convert individual element (string) to float
            // strtof is preferred to atof for robustness/error handling
            char *endptr;
            data[i * ncols + j] = strtof(token, &endptr);
            
            // Check whether conversion was successful 
            if (token == endptr) {
                fprintf(stderr, "Error parsing value at row %d, column %d\n", i+1, j+1);
                fclose(file);
                free(line);
                return;
            }

            // Get the next character
            token = strtok_r(NULL, ",", &saveptr);
        }
    }
    fclose(file);
    free(line);
    printf("Loaded csv data from %s: %d rows, %d columns\n", file_location, nrows, ncols);
}

void print_matrix(const float* matrix, int rows, int cols, const char* name){
    printf("Matrix %s showing top-left 8x8\n", name);
    for (int i = 0; i < 8 && i < rows; i++){
        for (int j = 0; j < 8 && j < cols; j++){
            printf("%7.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int all_close(float *v1, float *v2, int size){
    float tol = 0.001;
    for (int i = 0; i < size; i++){
        if( fabs(v1[i] - v2[i])/v1[i] > tol) {
            return 0;
        }
    }
    return 1;
}

void init_layer(Linear* layer, int in_features, int out_features){
    layer->W = (float*) malloc(in_features * out_features * sizeof(float));
    layer->b = (float*) malloc(out_features * sizeof(float));
    layer->input_size = in_features;
    layer->output_size = out_features;
}

#ifndef TEST

int main(int argc, char *argv[]) {

    const char *model_path = (argc > 1) ? argv[1] : "/tmp/ffwd.bin";
    const char *features_path = (argc > 2) ? argv[2] : "/tmp/CaliforniaHousing/features.csv";
    const char *output_path = (argc > 3) ? argv[3] : "/tmp/predictions.txt";

    int B = 16;        // batch dim
    int C_in = 8;      // input feature size
    int C = 32;        // hidden feature size

    // Allocate memory for model
    Linear layer_in;
    Linear layer1;
    Linear layer2;
    Linear layer3;
    Linear layer_out;

    init_layer(&layer_in, C_in, C);
    init_layer(&layer1, C, C);
    init_layer(&layer2, C, C);
    init_layer(&layer3, C, C);
    init_layer(&layer_out, C, 1);
    
    FILE *model_file = fopen(model_path, "rb");
    if (model_file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", model_path);
        return 1;
    }
    printf("Loaded model from %s\n", model_path);

    fread(layer_in.W, sizeof(float), C * C_in, model_file);
    fread(layer_in.b, sizeof(float), C, model_file);

    fread(layer1.W, sizeof(float), C * C, model_file);
    fread(layer1.b, sizeof(float), C, model_file);
    
    fread(layer2.W, sizeof(float), C * C, model_file);
    fread(layer2.b, sizeof(float), C, model_file);

    fread(layer3.W, sizeof(float), C * C, model_file);
    fread(layer3.b, sizeof(float), C, model_file);

    fread(layer_out.W, sizeof(float), 1 * C, model_file);
    fread(layer_out.b, sizeof(float), 1, model_file);
    fclose(model_file);

    // Read features into memory
    int nrows = 20640;
    int ncols = C_in;
    float *features = malloc(nrows * C_in * sizeof(float));
    if (features == NULL) {
        fprintf(stderr, "Failed to allocated memory for features\n");
        return 1;
    }
    read_csv(features, features_path, nrows, ncols);    
    print_matrix(features, B, C_in, "Features");

    // Make predictions
    float *predictions = malloc(nrows * 1 * sizeof(float));
    if (predictions == NULL) {
        fprintf(stderr, "Failed to allocated memory for predictions\n");
        free(features);
        return 1;
    }
    forward(features, nrows, C_in, &layer_in, &layer1, &layer2, &layer3, &layer_out, predictions);
    print_matrix(predictions, nrows, 1, "Predictions");

    // Save predictions in text format
    FILE *pred_file = fopen(output_path, "w");
    if (pred_file == NULL) {
        fprintf(stderr, "Error opening output file: %s\n", output_path);
        free(features);
        free(predictions);
        return 1;
    }
    for (int i = 0; i < nrows; i++) {
        fprintf(pred_file, "%f\n", predictions[i]);
    }
    printf("Wrote predictions to %s.\n", output_path);
    fclose(pred_file);

    // Clean up
    free(layer_in.W); free(layer_in.b);
    free(layer1.W); free(layer1.b);
    free(layer2.W); free(layer2.b);
    free(layer3.W); free(layer3.b);
    free(layer_out.W); free(layer_out.b);
    free(features); free(predictions);
    return 0;
}
#endif