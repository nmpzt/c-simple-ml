// demo: import model from demo0 and test it

#include <stdio.h>
#include <assert.h>

#define STB_LEAKCHECK_IMPLEMENTATION
#include "include/stb_leakcheck.h"

#define MLP_H_IMPLEMENTATION
#include "../mlp.h"

int main(void) {
  MLP mynn;
  assert(!MLP_load(&mynn, "models/xor_model.bin") && "Could not load model");

  float inputs[] = {
    0.f, 0.f,
    1.f, 0.f,
    0.f, 1.f,
    1.f, 1.f,
  };

  float output[1] = {0};
  for (int i = 0; i < 4; ++i) {
    MLP_set_inputs(&mynn, inputs + i);
    MLP_evaluate(&mynn);
    MLP_get_outputs(&mynn, output);
    printf("%g\n", output[0]);
  }

  MLP_free(&mynn);

  stb_leakcheck_dumpmem();

  return 0;
}

