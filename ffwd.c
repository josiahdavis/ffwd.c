#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> 

// Define Linear as pointer to weights and biases"
typedef struct {
    float *W, *b;
    int input_size, output_size;
} Linear;

// Define FeedForward as a pointer to linear objects
typedef struct {
    Linear *layer;
    int n_layers;
} FeedForward;

void init_layer(Linear* layer, int in_features, int out_features){
    layer->W = (float*) malloc(in_features * out_features * sizeof(float));
    layer->b = (float*) malloc(out_features * sizeof(float));
    layer->input_size = in_features;
    layer->output_size = out_features;
}

// Initialize model
FeedForward* create_model(int n_layers, int *layer_sizes){
    if (n_layers  < 2) {
        fprintf(stderr, "Network must have at least one input and output layers\n");
        return NULL;
    }

    // Instantiate model
    // malloc is needed to persist beyond function scope even through memory is not known yet.
    FeedForward *ffwd = (FeedForward*)malloc(sizeof(FeedForward));
    ffwd->n_layers = n_layers;
    ffwd->layer = (Linear*)malloc(ffwd->n_layers * sizeof(Linear));
    if (!ffwd->layer){
        perror("Failed to allocated network layers");
        free(ffwd);
        return NULL;
    }

    // Allocate memory for each layer
    for (int i = 0; i < ffwd->n_layers; i++){

        // Using & "address-of" operator because we want to pass pointer to the linear object 
        // not linear object itself
        init_layer(&ffwd->layer[i], layer_sizes[i], layer_sizes[i+1]);

        // If we have any issue with initialization then free entire network.
        if (!ffwd->layer[i].W || !ffwd->layer[i].b) {
            for (int j = 0; j < i; j++){
                free(ffwd->layer[j].W);
                free(ffwd->layer[j].b);
            }
            free(ffwd->layer);
            free(ffwd);
            return NULL;
        }
    }
    return ffwd;
}

// Remove model from memory
void free_model(FeedForward *ffwd){
    if (ffwd->layer){
        for (int i = 0; i < ffwd->n_layers; i++){
            free(ffwd->layer[i].W);
            free(ffwd->layer[i].b);
        }
        free(ffwd->layer);
    }
    free(ffwd);
}

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
    
    // Initialize bias in output
    if (bias != NULL) {
        for (int b = 0; b < B; b++){
            for (int c = 0; c < C_out; c++) {
                out[b * C_out + c] = bias[c];
            }
        }
    } else {
        memset(out, 0, B * C_out * sizeof(float));
    }

    // Perform matmul with (cache-aware) loop reodering
    for (int b = 0; b < B; b++){
        for (int i = 0; i < C_in; i++){
            // Small memory access optimization
            float x_val = x[b * C_in + i];
            for (int c = 0; c < C_out; c++){
                // formula for indexing into a matrix to get correct element:
                //      row index * column width + column index
                //          out = b * C_out + c
                //            x = b * C_in + i
                //           Wx = i * C_out + c
                out[b * C_out + c] += x_val * Wx[i * C_out + c];
            }
        }
    }
}

void relu(float *v, int size){
    for (int i = 0; i < size; i++){
        v[i] = (v[i] > 0) ? v[i] : 0;
    }
}

void forward(float *x, int B, int C_in, FeedForward *ffwd, float *out){
    // Assumes you have at least two linear layers
    float *current_input = x;
    float *current_output = NULL;
    for (int i = 0; i < ffwd->n_layers; i++){
        int input_size = ffwd->layer[i].input_size;
        int output_size = ffwd->layer[i].output_size;
        
        // Alocate memory for intermediate output
        if (i < ffwd->n_layers - 1){
            current_output = (float*)malloc(B * output_size * sizeof(float));
            if (current_output == NULL) {
                perror("Memory allocation failed in forward pass\n");
                exit(1);
            }
        } else {
            current_output = out;
        }

        // Perform matmul
        matmul_bias(current_input, ffwd->layer[i].W, ffwd->layer[i].b, current_output, B, input_size, output_size);

        // Apply ReLU
        if (i < (ffwd->n_layers - 1)){
            relu(current_output, B * ffwd->layer[i].output_size);
        }

        // Free previous input (but not original input x)
        if (i > 0) {
            free(current_input);
        }
        current_input = current_output;
    }
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
    size_t len = 0;
    ssize_t read;
    
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
        if( fabs(v1[i] - v2[i])/v2[i] > tol) {
            printf("%7.4f != %7.4f \n", v1[i], v2[i]);
            return 0;
        }
    }
    return 1;
}

#ifndef TEST

int main(int argc, char *argv[]) {
    clock_t start = clock();
    // ---------------------------
    // Setup
    // ---------------------------
    const char *model_path = (argc > 1) ? argv[1] : "/tmp/ffwd.bin";
    const char *features_path = (argc > 2) ? argv[2] : "/tmp/CaliforniaHousing/features.csv";
    const char *output_path = (argc > 3) ? argv[3] : "/tmp/predictions.txt";

    int B = 1024;        // batch dim
    int C_in = 8;        // input feature size
    int C = 1024;        // hidden feature size

    int layer_sizes[] = {C_in, C, C, C, C, 1};
    // Standard way of getting the length of an array in C
    int n_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]) - 1;

    // Instantiate model
    FeedForward *ffwd = create_model(n_layers, layer_sizes);
    if (ffwd == NULL){
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    // ---------------------------
    // Load model weights
    // ---------------------------
    FILE *model_file = fopen(model_path, "rb");
    if (model_file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", model_path);
        free_model(ffwd);
        return 1;
    }
    printf("Loaded model from %s\n", model_path);
    for (int i = 0; i < n_layers; i++) {
        size_t W_size = ffwd->layer[i].input_size * ffwd->layer[i].output_size;
        size_t b_size = ffwd->layer[i].output_size;

        // Use "." to access struct element, in this case these are float* elements
        // layer[i].W is already a pointer (to the weight array), which is what fread expects
        if (fread(ffwd->layer[i].W, sizeof(float), W_size, model_file) != W_size || 
            fread(ffwd->layer[i].b, sizeof(float), b_size, model_file) != b_size) {
                fprintf(stderr, "Error reading model parameters for layer %d\n", i);
                fclose(model_file);
                free_model(ffwd);
                return 1;
            }
    }
    fclose(model_file);

    // ---------------------------
    // Load features
    // ---------------------------
    int nrows = 20640;
    int ncols = C_in;
    float *features = malloc(nrows * C_in * sizeof(float));
    if (features == NULL) {
        fprintf(stderr, "Failed to allocated memory for features\n");
        free_model(ffwd);
        return 1;
    }
    read_csv(features, features_path, nrows, ncols);    
    print_matrix(features, B, C_in, "Features");

    // ---------------------------
    // Predict
    // ---------------------------
    float *predictions = malloc(nrows * 1 * sizeof(float));
    if (predictions == NULL) {
        fprintf(stderr, "Failed to allocated memory for predictions\n");
        free(features);
        free_model(ffwd);
        return 1;
    }

    clock_t fwd_start = clock();
    for (int batch_start = 0; batch_start < nrows; batch_start += B){
        // Adjust Batch size for last batch if needed
        int B_curr = (batch_start + B > nrows) ? (nrows - batch_start) : B;
        // Pointer arithmatic: features + (batch_start * C_in) = where in memory the current batch starts
        //      batch_start =  which sample to start from (0, 16, 32, 48, ...)
        //      C_in = number of features per sample
        //      batch_start * C_in = how many float values to skip
        forward(features + (batch_start * C_in), B_curr, C_in, ffwd, predictions + batch_start);
    }
    double fwd_time = ((double)(clock() - fwd_start))/CLOCKS_PER_SEC;
    printf("forward pass completed in %7.4f seconds \n", fwd_time);

    print_matrix(predictions, nrows, 1, "Predictions");

    // ---------------------------
    // Save predictions to text file
    // ---------------------------
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

    // ---------------------------
    // Clean up
    // ---------------------------
    free_model(ffwd);
    free(features); 
    free(predictions);
    double prog_time = ((double)(clock() - fwd_start))/CLOCKS_PER_SEC;
    printf("predictions completed in %7.4f seconds \n", prog_time);
    return 0;
}
#endif