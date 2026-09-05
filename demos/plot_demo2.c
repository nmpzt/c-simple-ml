#include <stdio.h>
#include <stdlib.h>
#include "../mlp.h"

#define RESOLUTION 1080

int main(void) {
  puts("Initialising...");
  MLP mymlp;
  if (MLP_load(&mymlp, "models/spiral_model.bin") != 0) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
  }

  FILE *f = fopen("output.ppm", "w");
  if (!f) {
    perror("Failed to open output file");
    MLP_free(&mymlp);
    return 1;
  }

  fprintf(f, "P3\n%d %d\n255\n", RESOLUTION, RESOLUTION);

  float min_val = -2.0f;
  float max_val = 2.0f;

  puts("Drawing...");
  for (int y = 0; y < RESOLUTION; ++y) {
    float fy = max_val - (float)y / (RESOLUTION - 1.0f) * (max_val - min_val);
    for (int x = 0; x < RESOLUTION; ++x) {
      float fx = min_val + (float)x / (RESOLUTION - 1.0f) * (max_val - min_val);

      float inputs[2] = {fx, fy};
      MLP_set_inputs(&mymlp, inputs);
      MLP_evaluate(&mymlp);

      float output[1];
      MLP_get_outputs(&mymlp, output);

      int red = (int)(output[0] * 255.0f);
      int blue = (int)((1.0f - output[0]) * 255.0f);
      int green = 120;

      fprintf(f, "%d %d %d\n", red, green, blue);
    }
  }

  fclose(f);
  MLP_free(&mymlp);
  puts("Done");
  return 0;
}
