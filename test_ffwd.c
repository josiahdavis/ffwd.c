#define TEST
#include "ffwd.c"

int main(){
    int B = 16;        // batch dim
    int C_in = 8;      // input feature size
    int C = 32;        // hidden feature size

    // Allocate for model
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
    
    // Read model into memory
    FILE *model_file = fopen("/tmp/ffwd.bin", "rb");
    if (model_file == NULL) {
        printf("Error opening file\n");
    }

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
    forward(batch_features, B, C_in, &layer_in, &layer1, &layer2, &layer3, &layer_out, out);
    
    // Print out info
    // print_matrix(&layer_out->W, C, 1, "W_out");
    print_matrix(batch_features, B, C_in, "batch_features");
    print_matrix(batch_labels, B, 1, "batch_labels");
    print_matrix(out_expected, B, 1, "out_expected");
    print_matrix(out, B, 1, "output");

    // Check that we are getting expected results
    int equal = all_close(out, out_expected, B * 1);
    if (equal == 1) printf("✅ SUCCESS\n");
    else if (equal == 0) printf("❌ ERROR\n");

    // Clean up
    free(layer_in.W); free(layer_in.b);
    free(layer1.W); free(layer1.b);
    free(layer2.W); free(layer2.b);
    free(layer3.W); free(layer3.b);
    free(layer_out.W); free(layer_out.b);

    free(batch_features);
    free(batch_labels);
    free(out_expected);
    free(out);

    return 0;
}