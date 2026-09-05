// demo2.c classifying spirals

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "../mlp.h"

#define NUM_SAMPLES_PER_CLASS 250
#define TOTAL_SAMPLES (2 * NUM_SAMPLES_PER_CLASS)

void generate_spiral_dataset(float*** out_inputs, float*** out_targets) {
  float** inputs = (float**)malloc(TOTAL_SAMPLES * sizeof(float*));
  float** targets = (float**)malloc(TOTAL_SAMPLES * sizeof(float*));

  for (int i = 0; i < TOTAL_SAMPLES; ++i) {
    inputs[i] = (float*)malloc(2 * sizeof(float));
    targets[i] = (float*)malloc(1 * sizeof(float));
  }

  for (int class_idx = 0; class_idx < 2; ++class_idx) {
    for (int i = 0; i < NUM_SAMPLES_PER_CLASS; ++i) {
      int idx = class_idx * NUM_SAMPLES_PER_CLASS + i;
      float r = (float)i / (float)NUM_SAMPLES_PER_CLASS * 1.5f;
      float t = (float)i / (float)NUM_SAMPLES_PER_CLASS * 2.5f * PI_F + (class_idx * PI_F);
      
      // add noise
      float noise_x = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
      float noise_y = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;

      inputs[idx][0] = r * sinf(t) + noise_x;
      inputs[idx][1] = r * cosf(t) + noise_y;
      targets[idx][0] = (float)class_idx;
    }
  }

  *out_inputs = inputs;
  *out_targets = targets;
}

void free_dataset(float** inputs, float** targets) {
  for (int i = 0; i < TOTAL_SAMPLES; ++i) {
    free(inputs[i]);
    free(targets[i]);
  }
  free(inputs);
  free(targets);
}

void log_function(MLP* nn, size_t epoch_num, float avg_loss) {
  printf("Epoch %zu - avg loss (MSE): %.6f\n", epoch_num + 1, avg_loss);
}

int main(int argc, char** argv) {
  if (argc == 1) {
    printf("Usage: %s <seed>", argv[0]);
    return 0;
  }
  puts("Initialising...");

  unsigned int seed = atoi(argv[1]);
  srand((unsigned int)time(NULL));

  float** dataset_inputs = NULL;
  float** dataset_targets = NULL;
  generate_spiral_dataset(&dataset_inputs, &dataset_targets);

  Topology mytopo;
  Topology_init(&mytopo, 4, (uint16_t[]) {2, 16, 16, 1});

  MLP mymlp;
  MLP_initialise(&mymlp, mytopo, ACT_tanh, ACT_logistic);

  MLP_populate(&mymlp, INIT_He, 42);

  puts("Trainign...");
  size_t epochs = 2000;
  float learning_rate = 0.05f;

  clock_t start_time = clock();

  MLP_train_logged(&mymlp, dataset_inputs, dataset_targets,
    TOTAL_SAMPLES, epochs, learning_rate, LOSS_MSE, 200, log_function);
  clock_t end_time = clock();
  printf("Training completed in %.2f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  int correct = 0;
  float output[1];
  for (int i = 0; i < TOTAL_SAMPLES; ++i) {
      MLP_set_inputs(&mymlp, dataset_inputs[i]);
      MLP_evaluate(&mymlp);
      MLP_get_outputs(&mymlp, output);
      int predicted_class = (output[0] >= 0.5f) ? 1 : 0;
      if ((float)predicted_class == dataset_targets[i][0]) {
          correct++;
      }
  }
  printf("Training Accuracy: %.2f%%\n", ((float)correct / (float)TOTAL_SAMPLES) * 100.0f);

  puts("Saving...");
  assert(!MLP_save(&mymlp, "models/spiral_model.bin") && "Failed to save spiral model");

  MLP_free(&mymlp);
  free_dataset(dataset_inputs, dataset_targets);
  puts("Done.");

  return 0;
}
