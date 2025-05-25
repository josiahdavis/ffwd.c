#define TEST
#include "ffwd.c"

int main(int argc, char *argv[]) {

    const char *model_path = (argc > 1) ? argv[1] : "/tmp/ffwd.bin";
    const char *features_path = (argc > 2) ? argv[2] : "/tmp/CaliforniaHousing/features.csv";
    const char *output_path = (argc > 3) ? argv[3] : "/tmp/predictions.txt";

    int B = 16;        // batch dim
    int C_in = 8;      // input feature size
    int C = 32;        // hidden feature size

    // Allocate for model
    // Instantiate and allocate memory for model
    int layer_sizes[] = {C_in, C, C, C, C, 1};
    // Standard way of getting the length of an array in C
    int n_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]) - 1;
    FeedForward *ffwd = create_model(n_layers, layer_sizes);
    if (ffwd == NULL){
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    // Load model weights
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

        // fread returns number of elements read successfully from file.
        if (fread(ffwd->layer[i].W, sizeof(float), W_size, model_file) != W_size || 
            fread(ffwd->layer[i].b, sizeof(float), b_size, model_file) != b_size) {
                fprintf(stderr, "Error reading model parameters for layer %d\n", i);
                fclose(model_file);
                free_model(ffwd);
                return 1;
            }
    }
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
    forward(batch_features, B, C_in, ffwd, out);
    
    // Print out info
    print_matrix(batch_features, B, C_in, "batch_features");
    print_matrix(batch_labels, B, 1, "batch_labels");
    print_matrix(out_expected, B, 1, "out_expected");
    print_matrix(out, B, 1, "output");

    // Check that we are getting expected results
    int equal = all_close(out, out_expected, B * 1);
    if (equal == 1) printf("✅ SUCCESS\n");
    else if (equal == 0) printf("❌ ERROR\n");

    // Clean up

    free(batch_features);
    free(batch_labels);
    free(out_expected);
    free(out);
    free_model(ffwd);
    return 0;
}