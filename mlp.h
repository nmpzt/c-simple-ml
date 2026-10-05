#pragma once

#include <stddef.h> // size_t
#include <stdlib.h> // malloc & free
#include <stdint.h> // for smaller numbers
#include <stdio.h>
#include <math.h>   // math.h

// MLPE stands for MLP error
typedef enum {
  MLPE_SUCCESS = 0,
  MLPE_NULL_PTR,     // void pointer passed into function
  MLPE_MALLOC,
  MLPE_FILE_ERROR,
  MLPE_WHAT,         // unknown i guess
} MLPE_CODE;

#ifndef PI_F
#define PI_F 3.14159265358979323846f
#endif

#ifndef PI2_F
#define PI2_F 6.28318548202514648438f
#endif

static float random_standard_normal(void) {
  // Z1 = sqrt(-2 ln U1) * cos 2pi U2
  float u1;
  do {
    u1 = ((float)rand() + 1.f) / ((float)RAND_MAX + 2.f);
  } while (u1 <= 0.0f || u1 > 1.0f);

  float u2 = ((float)rand() + 1.f) / ((float)RAND_MAX + 2.f);

  return sqrtf(-2.f * logf(u1)) * cosf(PI2_F * u2);
}

typedef enum {
  ACT_linear,
  ACT_relu,
  ACT_leaky_relu,
  ACT_logistic,
  ACT_tanh,
} Activation;

typedef enum {
  INIT_zero,
  INIT_Xavier,
  INIT_He,
} Initialisation;

typedef enum {
  LOSS_MSE,
  LOSS_ASE,
  LOSS_binary_cross_entropy,
} Loss_func;

// ACTivation functions

#ifndef LEAKY_RELU_CONST
#define LEAKY_RELU_CONST 0.01
#endif

float ACT_linearf(float input) {
  return input; // i shouldn't have made this function
}

float ACT_reluf(float input) {
  return (input < 0) ? 0 : input;
}

float ACT_leaky_reluf(float input) {
  input *= (input < 0) ? LEAKY_RELU_CONST : 1;
  return input;
}

float ACT_logisticf(float input) {
  return 1.l / (1.l + expf(-input));
}

float ACT_tanhf(float input) {
  return tanhf(input);
}

// and their derivatives

float derv_linearf(float input) {
  return 1.l;
}

float derv_reluf(float input) {
  return (input < 0) ? 0 : 1;
}

float derv_leaky_reluf(float input) {
  return (input < 0) ? LEAKY_RELU_CONST : 1;
}

float derv_logisticf(float activated_value) {
  return activated_value * (1 - activated_value);
}

float derv_tanhf(float activated_value) {
  return 1 - activated_value * activated_value;
}

// topology

typedef struct {
  uint16_t  num_layers;
  uint16_t* layer_sizes;
  size_t*   weight_offsets; // cached start index of weights for each layer
  size_t*   val_offsets;    // cached start index of biases/values for each layer
} Topology;
// e.g. {2, 3, 1} means the layers have 2, 3, 1 neurons respectively

MLPE_CODE Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons) {
  if (!topo || !neurons) return MLPE_NULL_PTR;
  topo->num_layers = layers;
  topo->layer_sizes = (uint16_t*) malloc(layers * sizeof(uint16_t));

  if (!topo->layer_sizes) return MLPE_MALLOC;

  topo->weight_offsets = (size_t*) malloc(layers * sizeof(size_t));

  if (!topo->weight_offsets) {
    free(topo->layer_sizes);
    return MLPE_MALLOC;
  }

  topo->val_offsets = (size_t*) malloc(layers * sizeof(size_t));
  if (!topo->val_offsets) {
    free(topo->layer_sizes);
    free(topo->weight_offsets);
    return MLPE_MALLOC;
  }

  size_t w_acc = 0;
  size_t v_acc = 0;

  for (uint16_t i = 0; i < layers; ++i) {
    topo->layer_sizes[i] = neurons[i];

    topo->weight_offsets[i] = w_acc;
    if (i < layers - 1) {
      w_acc += (size_t)neurons[i] * neurons[i + 1];
    }

    // store the running accumulator as the offset for layer i
    topo->val_offsets[i] = v_acc;
    v_acc += neurons[i];
  }

  return MLPE_SUCCESS;
}

MLPE_CODE Topology_free(Topology* topo) {
  if (!topo) return MLPE_NULL_PTR;
  topo->num_layers = 0;
  free(topo->layer_sizes);
  free(topo->weight_offsets);
  free(topo->val_offsets);
  return MLPE_SUCCESS;
}

// neural network

typedef struct {
  float*  weights;
  float*   biases;
  float*   values;    // ACTivated values a
  float* raw_sums;    // z for backprop

  float*  values_deltas;
  float* weights_deltas;
  float*  biases_deltas;

  float*  memory_pool; // all the data live here

  Topology topo;

  Activation hidden_layer_activation;
  Activation output_layer_activation;
} MLP;

/**
 * Initialise a neural network using a single memory pool allocation
 * Note: This only allocates memory,
 * To populate the network w/ random parameters, use the MLP_populate() function instead
 */
MLPE_CODE MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act) {
  if (!input) return MLPE_NULL_PTR;
  input->topo = topo;
  input->hidden_layer_activation = hidden_act;
  input->output_layer_activation = output_act;

  size_t weights_size = 0;
  size_t all_size = 0;

  // calculate weights size (between adjacent layers)
  for (uint16_t i = 0; i < topo.num_layers - 1; ++i) {
    weights_size += (size_t)topo.layer_sizes[i] * topo.layer_sizes[i + 1];
  }

  for (uint16_t i = 0; i < topo.num_layers; ++i) {
    all_size += topo.layer_sizes[i];
  }

  size_t total_floats = (2 * weights_size) + (5 * all_size);
  input->memory_pool = (float*) malloc(total_floats * sizeof(float));
  if (!input->memory_pool) return MLPE_MALLOC;

  // slice the memory pool into individual internal pointers
  input->weights        = input->memory_pool;
  input->weights_deltas = input->weights        + weights_size;
  input->biases         = input->weights_deltas + weights_size;
  input->values         = input->biases         + all_size;
  input->raw_sums       = input->values         + all_size;
  input->values_deltas  = input->raw_sums       + all_size;
  input->biases_deltas  = input->values_deltas  + all_size;

  return MLPE_SUCCESS;
}

// index lookup helpers

// ? Should we make these macros?

// weights index lookup using cached offsets
static inline size_t weights_index(size_t layer, size_t target, size_t source, const Topology* topo) {
  // weights are stored as [source][target]:
  return topo->weight_offsets[layer] + (source * topo->layer_sizes[layer + 1] + target);
}

// value/bias index lookup using cached offsets
static inline size_t val_index(size_t layer, size_t ind, const Topology* topo) {
  // for layer > 0, val_offsets[layer] points straight to the start of that layer's block
  return topo->val_offsets[layer] + ind;
}
/**
 * Populates the neural network using the chosen initialisation
 */
MLPE_CODE MLP_populate(MLP* input, Initialisation initialisation_method, unsigned int seed) {
  if (!input) return MLPE_NULL_PTR;
  // seed
  srand(seed);

  Topology* topo = &(input->topo);

  // initialize biases to zero (standard practice across all three methods)
  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
    all_size += topo->layer_sizes[i];
  }
  for (size_t i = 0; i < all_size; ++i) {
    input->biases[i] = 0.0;
  }

  // initialize weights
  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
    weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  if (initialisation_method == INIT_zero) {
    for (size_t i = 0; i < weights_size; ++i) {
      input->weights[i] = 0.0;
    }
  } 
  else if (initialisation_method == INIT_Xavier) {
    // Xavier initialization: normal distribution with stddev = sqrt(2 / (n_in + n_out))
    for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
      float n_in  = (float)topo->layer_sizes[l];
      float n_out = (float)topo->layer_sizes[l + 1];
      float stddev = sqrtf(2.f / (n_in + n_out));

      for (size_t target = 0; target < topo->layer_sizes[l + 1]; ++target) {
        for (size_t source = 0; source < topo->layer_sizes[l]; ++source) {
          size_t idx = weights_index(l, target, source, topo);
          input->weights[idx] = random_standard_normal() * stddev;
        }
      }
    }
  } 
  else if (initialisation_method == INIT_He) {
    // He initialization: normal distribution with stddev = sqrt(2 / n_in)
    for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
      float n_in = (float)topo->layer_sizes[l];
      float stddev = sqrtf(2.f / n_in);

      for (size_t target = 0; target < topo->layer_sizes[l + 1]; ++target) {
        for (size_t source = 0; source < topo->layer_sizes[l]; ++source) {
          size_t idx = weights_index(l, target, source, topo);
          input->weights[idx] = random_standard_normal() * stddev;
        }
      }
    }
  }

  return MLPE_SUCCESS;
}

void MLP_free(MLP* input) {
  free(input->memory_pool);
  input->memory_pool = NULL;
  input->weights = NULL;
  input->weights_deltas = NULL;
  input->biases = NULL;
  input->values = NULL;
  input->raw_sums = NULL;
  input->values_deltas = NULL;
  input->biases_deltas = NULL;

  Topology_free( &(input->topo) );
}

/**
 * Sets the values of the first layer of the neural network
 * Otherwise there will be a demon at your doorstep tonight.
 */
MLPE_CODE MLP_set_inputs(MLP* mlp, float values[]) {
  if (!mlp) return MLPE_NULL_PTR;
  uint16_t firstlayer_size = mlp->topo.layer_sizes[0];
  for (uint16_t i = 0; i < firstlayer_size; ++i) {
    mlp->values[val_index(0, i,  &(mlp->topo) )] = values[i];
  }

  return MLPE_SUCCESS;
}

/**
 * Set rop's value to that of the network's last layer
 */
MLPE_CODE MLP_get_outputs(MLP* mlp, float* rop) {
  if (!mlp) return MLPE_NULL_PTR;
  const size_t last_layer_index = mlp->topo.num_layers - 1;
  const size_t last_layer_size = mlp->topo.layer_sizes[last_layer_index];

  for (size_t i = 0; i < last_layer_size; ++i) {
    rop[i] = mlp->values[val_index(last_layer_index, i, &(mlp->topo) )];
  }

  return MLPE_SUCCESS;
}

// helper function to apply the correct ACTivation function based on the enum
static float apply_activation(Activation act, float input) {
  switch (act) {
    case ACT_linear:       return ACT_linearf(input);
    case ACT_relu:         return ACT_reluf(input);
    case ACT_leaky_relu:   return ACT_leaky_reluf(input);
    case ACT_logistic:     return ACT_logisticf(input);
    case ACT_tanh:         return ACT_tanhf(input);
    default:               return input;
  }
}

MLPE_CODE MLP_evaluate(MLP* mlp) {
  if (!mlp) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);

  // iterate through each layer, starting from the first hidden layer (layer 1)
  for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
    uint16_t current_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];
    Activation act = (l == topo->num_layers - 2) ? mlp->output_layer_activation 
                                                 : mlp->hidden_layer_activation;

    // initialize the raw sums for the next layer with biases
    for (size_t j = 0; j < next_size; ++j) {
      size_t next_v_idx = val_index(l + 1, j, topo);
      mlp->raw_sums[next_v_idx] = mlp->biases[next_v_idx];
      
      // add weighted inputs from the current layer
      for (size_t i = 0; i < current_size; ++i) {
          size_t curr_v_idx = val_index(l, i, topo);
          size_t w_idx = weights_index(l, j, i, topo); // layer, target, source
          
          mlp->raw_sums[next_v_idx] += mlp->weights[w_idx] * mlp->values[curr_v_idx];
      }
      
      // apply the activation function to get the activated value
      mlp->values[next_v_idx] = apply_activation(act, mlp->raw_sums[next_v_idx]);
    }
  }

  return MLPE_SUCCESS;
}

static float apply_activation_derivative(Activation act, float activated_val) {
  switch (act) {
    case ACT_linear:       return derv_linearf(activated_val);
    case ACT_relu:         return derv_reluf(activated_val);
    case ACT_leaky_relu:   return derv_leaky_reluf(activated_val);
    case ACT_logistic:     return derv_logisticf(activated_val);
    case ACT_tanh:         return derv_tanhf(activated_val);
    default:               return 1.0;
  }
}

/**
 * Computes backprop for a single training sample given the target outputs.
 */
MLPE_CODE MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func) {
  if (!mlp || !targets) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);
  uint16_t L = topo->num_layers - 1; // index of output layer
  size_t output_size = topo->layer_sizes[L];

  // compute output layer deltas based on the choice of loss function
  for (size_t j = 0; j < output_size; ++j) {
    size_t v_idx = val_index(L, j, topo);
    float a = mlp->values[v_idx];
    float t = targets[j];
    float derivative = apply_activation_derivative(mlp->output_layer_activation, a);

    // derivative of loss w.r.t activated output (a - t for MSE, etc.)
    float loss_derv = 0.0;
    if (loss_func == LOSS_MSE || loss_func == LOSS_ASE) {
      loss_derv = (a - t); // factor of 2 can be absorbed into learning rate
    } else if (loss_func == LOSS_binary_cross_entropy) {
      // prevent division by zero
      float eps = 1e-15;
      float clamped_a = fmax(eps, fmin(1.0 - eps, a));
      loss_derv = (clamped_a - t) / (clamped_a * (1.0 - clamped_a));
    }

    mlp->values_deltas[v_idx] = loss_derv * derivative;
  }

  // propagate deltas backward through the hidden layers
  for (int16_t l = (int16_t)L - 1; l >= 0; --l) {
    uint16_t curr_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];

    for (size_t i = 0; i < curr_size; ++i) {
      size_t curr_v_idx = val_index(l, i, topo);
      float error_sum = 0.0;

      for (size_t j = 0; j < next_size; ++j) {
        size_t next_v_idx = val_index(l + 1, j, topo);
        size_t w_idx = weights_index(l, j, i, topo);
        error_sum += mlp->weights[w_idx] * mlp->values_deltas[next_v_idx];
      }

      float a = mlp->values[curr_v_idx];
      Activation act = mlp->hidden_layer_activation;
      
      // for hidden layers, multiply by activation derivative w.r.t raw sum / activated value
      mlp->values_deltas[curr_v_idx] = error_sum * apply_activation_derivative(act, a);
    }
  }

  // acumulate gradients for weights and biases
  for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
    uint16_t curr_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];

    for (size_t j = 0; j < next_size; ++j) {
      size_t next_v_idx = val_index(l + 1, j, topo);
      float next_delta = mlp->values_deltas[next_v_idx];

      // bias delta is directly the layer delta
      mlp->biases_deltas[next_v_idx] = next_delta;

      for (size_t i = 0; i < curr_size; ++i) {
        size_t curr_v_idx = val_index(l, i, topo);
        size_t w_idx = weights_index(l, j, i, topo);
        
        // weight gradient = input from previous layer * delta of target neuron
        mlp->weights_deltas[w_idx] = next_delta * mlp->values[curr_v_idx];
      }
    }
  }

  return MLPE_SUCCESS;
}

/**
 * Applies gradient descent to update weights and biases using the computed deltas
 */
MLPE_CODE MLP_update_weights(MLP* mlp, float learning_rate) {
  if (!mlp) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);

  // update weights
  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
    weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  for (size_t i = 0; i < weights_size; ++i) {
    mlp->weights[i] -= learning_rate * mlp->weights_deltas[i];
  }

  // update biases
  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
    all_size += topo->layer_sizes[i];
  }

  for (size_t i = 0; i < all_size; ++i) {
    // layer 0 biases dont receive updates since they have no incoming deltas
    mlp->biases[i] -= learning_rate * mlp->biases_deltas[i];
  }

  return MLPE_SUCCESS;
}

/**
 * Performs a single training step for a single input-target pair
 */
MLPE_CODE MLP_train_step(MLP* mlp, float inputs[], float targets[], float learning_rate, Loss_func loss_func) {
  if (!mlp || !inputs || !targets) return MLPE_NULL_PTR;
  MLPE_CODE ecode = MLPE_SUCCESS;
  // det the input values to the network's input layer
  ecode = MLP_set_inputs(mlp, inputs);
  if (ecode) return ecode;

  // perform the forward pass to compute predictions
  ecode = MLP_evaluate(mlp);
  if (ecode) return ecode;

  // perform backpropagation to compute gradients (deltas)
  ecode = MLP_backpropagate(mlp, targets, loss_func);
  if (ecode) return ecode;

  // update the weights and biases using gradient descent
  ecode = MLP_update_weights(mlp, learning_rate);
  if (ecode) return ecode;

  return MLPE_SUCCESS;
}

/**
 * wrapper for training
 */
MLPE_CODE MLP_train(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func) {
  if (!mlp || !dataset_inputs || !dataset_targets || !*dataset_inputs || !*dataset_targets) return MLPE_NULL_PTR;

  MLPE_CODE ecode = MLPE_SUCCESS;
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < num_samples; ++i) {
      ecode = MLP_train_step(mlp, dataset_inputs[i], dataset_targets[i], learning_rate, loss_func);
      if (ecode) return ecode;
    }
  }

  return MLPE_SUCCESS;
}

// NOTE: This function is slower than its unlogged equivalent due to the overhead
MLPE_CODE MLP_train_logged(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func,
              size_t log_step,
              void (*logging_func)(MLP* nn, size_t epoch_num, float avg_loss)) {
  if (!mlp || !dataset_inputs || !dataset_targets || !*dataset_inputs || !*dataset_targets || !logging_func) return MLPE_NULL_PTR;

  size_t output_size = mlp->topo.layer_sizes[mlp->topo.num_layers - 1];
  float* output = (float*)malloc(output_size * sizeof(float));
  if (!output) return MLPE_MALLOC;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < num_samples; ++i) {
      MLPE_CODE ecode = MLP_train_step(mlp, dataset_inputs[i], dataset_targets[i], learning_rate, loss_func);
      if (ecode) return ecode;
    }

    if ((epoch + 1) % log_step == 0 && logging_func != NULL) {
      float total_loss = 0.0f;
      for (size_t i = 0; i < num_samples; ++i) {
        MLPE_CODE ecode = MLP_set_inputs(mlp, dataset_inputs[i]);
        if (ecode) return ecode;
        ecode = MLP_evaluate(mlp);
        if (ecode) return ecode;
        ecode = MLP_get_outputs(mlp, output);
        if (ecode) return ecode;

        for (size_t j = 0; j < output_size; ++j) {
          float a = output[j];
          float t = dataset_targets[i][j];
          if (loss_func == LOSS_MSE || loss_func == LOSS_ASE) {
            float diff = a - t;
            total_loss += diff * diff;
          } else if (loss_func == LOSS_binary_cross_entropy) {
            float eps = 1e-15f;
            float clamped_a = fmaxf(eps, fminf(1.0f - eps, a));
            total_loss -= (t * logf(clamped_a) + (1.0f - t) * logf(1.0f - clamped_a));
          }
        }
      }
      float avg_loss = total_loss / (float)num_samples;
      logging_func(mlp, epoch, avg_loss);
    }
  }
  free(output);

  return MLPE_SUCCESS;
}

/**
 * Saves the neural network structure, configurations, weights, and biases to a binary file.
 */
MLPE_CODE MLP_save(const MLP* mlp, const char* filename) {
  if (!mlp || !filename) return MLPE_NULL_PTR;
  FILE* file = fopen(filename, "wb");
  if (!file) {
    return MLPE_FILE_ERROR;
  }

  const Topology* topo = &(mlp->topo);

  // write metadata (layers count, and activation enums)
  fwrite(&(topo->num_layers), sizeof(uint16_t), 1, file);
  fwrite(&(mlp->hidden_layer_activation), sizeof(Activation), 1, file);
  fwrite(&(mlp->output_layer_activation), sizeof(Activation), 1, file);

  // write layer size array
  fwrite(topo->layer_sizes, sizeof(uint16_t), topo->num_layers, file);

  // calculate sizes for weights and total neurons (biases/values)
  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
      weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
      all_size += topo->layer_sizes[i];
  }

  // write weights and biases arrays
  fwrite(mlp->weights, sizeof(float), weights_size, file);
  fwrite(mlp->biases, sizeof(float), all_size, file);

  fclose(file);
  return MLPE_SUCCESS;
}

/**
 * Loads a neural network structure, weights, and biases from a binary file
 * Automatically handles memory allocation for the network and topology
 */
MLPE_CODE MLP_load(MLP* mlp, const char* filename) {
  if (!mlp || !filename) return MLPE_NULL_PTR;
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return MLPE_FILE_ERROR;
  }

  uint16_t num_layers = 0;
  Activation hidden_act, output_act;

  // read metadata
  if (fread(&num_layers, sizeof(uint16_t), 1, file) != 1 ||
    fread(&hidden_act, sizeof(Activation), 1, file) != 1 ||
    fread(&output_act, sizeof(Activation), 1, file) != 1) {
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  // read layer sizes
  uint16_t* layer_sizes = (uint16_t*)malloc(num_layers * sizeof(uint16_t));
  if (!layer_sizes) {
    fclose(file);
    return MLPE_MALLOC;
  }
  if (fread(layer_sizes, sizeof(uint16_t), num_layers, file) != num_layers) {
    free(layer_sizes);
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  // initialize topology and neural network memory structures
  Topology topo;
  MLPE_CODE ecode = Topology_init(&topo, num_layers, layer_sizes);
  if (ecode) {
    free(layer_sizes);
    fclose(file);
    return ecode;
  }
  free(layer_sizes); // Topology_init makes its own copy

  MLP_initialise(mlp, topo, hidden_act, output_act);

  // calculate expected array sizes
  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo.num_layers - 1; ++i) {
    weights_size += (size_t)topo.layer_sizes[i] * topo.layer_sizes[i + 1];
  }

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo.num_layers; ++i) {
    all_size += topo.layer_sizes[i];
  }

  // read weights and biases back into memory
  if (fread(mlp->weights, sizeof(float), weights_size, file) != weights_size ||
    fread(mlp->biases, sizeof(float), all_size, file) != all_size) {
    // cleanup if read fails mid-way
    MLP_free(mlp);
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  fclose(file);
  return MLPE_SUCCESS;
}
