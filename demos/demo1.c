// demo: import model from demo0 and test it

#include <stdio.h>
#include <assert.h>

#include "../mlp.h"

int main(void) {
  MLP mynn;
  assert(!MLP_load(&mynn, "models/xor_model.bin") && "Could not load model");

  float* inputs[] = {
    (float[]) {0.f, 0.f},
    (float[]) {1.f, 0.f},
    (float[]) {0.f, 1.f},
    (float[]) {1.f, 1.f},
  };

  float output[1] = {0};
  for (int i = 0; i < 4; ++i) {
    MLP_set_inputs(&mynn, inputs[i]);
    MLP_evaluate(&mynn);
    MLP_get_outputs(&mynn, output);
    printf("%g\n", output[0]);
  }

  return 0;
}

