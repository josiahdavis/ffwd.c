#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

void forward(float *x, int B, int C_in, int C, float* W_in, float* W1, float* W2, float* W3, float* W_out, 
    float* b_in, float* b1, float* b2, float* b3, float* b_out, float* out){
    
    // Input Projection, followed by Relu
    float* act1 = (float*) malloc(B * C * sizeof(float));
    matmul_bias(x, W_in, b_in, act1, B, C_in, C);
    relu(act1, B * C);

    // Linear, followed by Relu
    float* act2 = (float*) malloc(B * C * sizeof(float));
    matmul_bias(act1, W1, b1, act2, B, C, C);
    relu(act2, B * C);

    // Linear, followed by Relu
    float* act3 = (float*) malloc(B * C * sizeof(float));
    matmul_bias(act2, W2, b2, act3, B, C, C);
    relu(act3, B * C);

    // Linear, followed by Relu
    float* act4 = (float*) malloc(B * C * sizeof(float));
    matmul_bias(act3, W3, b3, act4, B, C, C);
    relu(act4, B * C);

    // Output projection
    matmul_bias(act4, W_out, b_out, out, B, C, 1);

}

void read_csv(float* data, const char* file_location, int nrows, int ncols) {
    // Poor man's csv file loader.
    // Assumes all data is numerical with no header. 
    // Stores in row-wise format.

    int max_line_len = 10000;
    char line[max_line_len];
    char *token;

    if (data == NULL) {
        perror("Memory allocation failed\n");
    }

    FILE *file = fopen(file_location, "r");
    if (file == NULL){
        perror("Missing file\n");
    }

    for (int i = 0; i < nrows; i++){
        fgets(line, max_line_len, file);
        for (int j = 0; j < ncols; j++){
            token = strtok(j == 0 ? line : NULL, ",");
            if (token == NULL){
                perror("Error parsing row");
                fclose(file);
            }
            data[i * ncols + j] = atof(token);
        }
    }
    fclose(file);
    printf("Loaded csv data\n");
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

int main() {
    int B = 16;        // batch dim
    int C_in = 8;      // input feature size
    int C = 32;        // hidden feature size

    // Allocate for model
    float* W_in = (float*) malloc(C * C_in * sizeof(float));
    float* b_in = (float*) malloc(C * sizeof(float));
    
    float* W1 = (float*) malloc(C * C * sizeof(float));
    float* b1 = (float*) malloc(C * sizeof(float));
    
    float* W2 = (float*) malloc(C * C * sizeof(float));
    float* b2 = (float*) malloc(C * sizeof(float));
    
    float* W3 = (float*) malloc(C * C * sizeof(float));
    float* b3 = (float*) malloc(C * sizeof(float));
    
    float* W_out = (float*) malloc(1 * C * sizeof(float));
    float* b_out = (float*) malloc(1 * sizeof(float));
    
    // Read model into memory
    FILE *model_file = fopen("/tmp/ffwd.bin", "rb");
    if (model_file == NULL) {
        printf("Error opening file\n");
    }

    fread(W_in, sizeof(float), C * C_in, model_file);
    fread(b_in, sizeof(float), C, model_file);

    fread(W1, sizeof(float), C * C, model_file);
    fread(b1, sizeof(float), C, model_file);

    fread(W2, sizeof(float), C * C, model_file);
    fread(b2, sizeof(float), C, model_file);

    fread(W3, sizeof(float), C * C, model_file);
    fread(b3, sizeof(float), C, model_file);

    fread(W_out, sizeof(float), 1 * C, model_file);
    fread(b_out, sizeof(float), 1, model_file);
    fclose(model_file);

    // Allocate for test data
    float* batch_features = (float*) malloc(B * C_in * sizeof(float));
    float* batch_labels = (float*) malloc(B * sizeof(float));
    float* out = (float*) malloc(1 * B * sizeof(float));
    float* out_expected = (float*) malloc(1 * B * sizeof(float));

    // Read test data into memory
    FILE *test_data = fopen("/tmp/data.bin", "rb");
    if (test_data == NULL) {
        printf("Error opening file\n");
    }
    fread(batch_features, sizeof(float), B * C_in, test_data);
    fread(batch_labels, sizeof(float), B, test_data);
    fread(out_expected, sizeof(float), B, test_data);
    fclose(test_data);

    // Run forward pass on test data
    forward(batch_features, B, C_in, C, W_in, W1, W2, W3, W_out, b_in, b1, b2, b3, b_out, out);
    
    // Print out info
    print_matrix(W_out, C, 1, "W_out");
    print_matrix(batch_features, B, C_in, "batch_features");
    print_matrix(batch_labels, B, 1, "batch_labels");
    print_matrix(out_expected, B, 1, "out_expected");
    print_matrix(out, B, 1, "output");

    // Check that we are getting expected results
    int equal = all_close(out, out_expected, B * 1);
    if (equal == 1) printf("✅ SUCCESS\n");
    else if (equal == 0) printf("❌ ERROR\n");

    // Read data into memory for inference
    int nrows = 20640;
    int ncols = C_in;
    float *features = malloc(nrows * C_in * sizeof(float));
    read_csv(features, "/tmp/CaliforniaHousing/features.csv", nrows, ncols);    
    print_matrix(features, B, C_in, "Features");

    // Make predictions
    float *predictions = malloc(nrows * 1 * sizeof(float));
    forward(features, B, C_in, C, W_in, W1, W2, W3, W_out, b_in, b1, b2, b3, b_out, predictions);
    print_matrix(predictions, B, 1, "Predictions");

    // Save predictions in text format
    FILE *pred_file = fopen("/tmp/predictions.txt", "w");
    if (pred_file == NULL) {
        printf("Error opening file\n");
    }
    for (int i = 0; i < nrows; i++) {
        fprintf(pred_file, "%f\n", predictions[i]);
    }
    printf("Wrote predictions to disk.\n");
    fclose(pred_file);

    // Clean up
    free(W_in);
    free(b_in);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(W3);
    free(b3);
    free(W_out);
    free(b_out);
    free(batch_features);
    free(batch_labels);
    free(out_expected);
    free(out);
    free(features);
    return 0;
}