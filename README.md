# Very Simple Machine Learning Library in C

A small single-header multilayer perceptron 'library' written in C (`mlp.h`). 

## Usage
To use this library, include `mlp.h` in your source files. In one C file, define `MLP_H_IMPLEMENTATION` before including the header to generate the implementation definitions:
```c
#define MLP_H_IMPLEMENTATION
#include "mlp.h"
```

When you compile it, link it against the math library (`-lm`), which this depends on.

## PRNG (`Xoshiro256**`)

The library uses the **Xoshiro256** (Xoshiro256starstar) pseudo-random number generator seeded using SplitMix64 for weight initialisation.
- **State struct**: `XS256_state`
- **Seeding function**: `void XS256_seed(XS256_state *s, uint64_t seed)`
- **Random generation**: `uint64_t XS256_rand(XS256_state *s)`

## Enums & Configurations

- **Activation functions**: `Activation`
    - `ACT_linear`: Linear.
    - `ACT_relu`: ReLU.
    - `ACT_leaky_relu`: Leaky ReLU.
    - `ACT_logistic`: Logistic / Sigmoid.
    - `ACT_tanh`: Tanh.

- **Weight Initializations**: `Initialisation`
    - `INIT_zero`: Zero Initialization.
    - `INIT_Xavier`: Xavier Initialization.
    - `INIT_He`: He Initialization.

- **Loss Functions**: `Loss_func`
    - `LOSS_MSE`: Mean Squared Error.
    - `LOSS_ASE`: Absolute Squared Error.
    - `LOSS_binary_cross_entropy`: Binary Cross-Entropy.

- **Error Codes**: `MLPE_CODE`
    - `MLPE_SUCCESS` (value 0): Success.
    - `MLPE_NULL_PTR`: Null pointer passed into function.
    - `MLPE_MALLOC`: Memory allocation failed.
    - `MLPE_FILE_ERROR`: Errors related to file opening, reading, or writing.
    - `MLPE_WHAT`: Unknown error (reserved).

- **Training Configuration**: `MLP_train_config`
    - `float learning_rate`
    - `Loss_func loss_func`
    - `size_t epochs`
    - `size_t log_step`
    - `void (*logging_func)(MLP* nn, size_t epoch_num, float avg_loss)`


## API Reference

### Topology Management

- `MLPE_CODE Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons)`: Initializes the network structure and caches weight/value offsets.
- `MLPE_CODE Topology_free(Topology* topo)`: Frees topology memory.

### Network Lifecycle

- `MLPE_CODE MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act)`: Allocates memory using a single contiguous memory pool for weights, biases, values, and gradient deltas.
- `MLPE_CODE MLP_populate(MLP* input, Initialisation initialisation_method, XS256_state* randstate)`: Populates weights using the specified initialization method and a pointer to an initialized Xoshiro256 state.
- `MLPE_CODE MLP_free(MLP* input)`: Frees all memory associated with the network pool and its topology.

### Execution & Training

- `MLPE_CODE MLP_set_inputs(MLP* mlp, float values[])`: Loads input features into the input layer.
- `MLPE_CODE MLP_evaluate(MLP* mlp)`: Performs forward propagation.
- `MLPE_CODE MLP_get_outputs(MLP* mlp, float* dest)`: Extracts predictions from the output layer.
- `MLPE_CODE MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func)`: Computes gradients via backpropagation for a single training sample.
- `MLPE_CODE MLP_update_weights(MLP* mlp, float learning_rate)`: Applies gradient descent to update weights and biases.
- `MLPE_CODE MLP_train_step(MLP* mlp, float inputs[], float targets[], const MLP_train_config* config)`: Runs a single forward pass, backpropagation, and weight update using a training configuration.
- `MLPE_CODE MLP_train(MLP* mlp, float** dataset_inputs, float** dataset_targets, size_t num_samples, const MLP_train_config* config)`: Standard batch training loop across multiple epochs using a training configuration.
- `MLPE_CODE MLP_train_logged(MLP* mlp, float** dataset_inputs, float** dataset_targets, size_t num_samples, const MLP_train_config* config)`: Training loop with periodic loss evaluation and logging callback support specified in the configuration.

### Saving & Loading

- `MLPE_CODE MLP_save(const MLP* mlp, const char* filename)`: Saves a network to disk.
- `MLPE_CODE MLP_load(MLP* mlp, const char* filename)`: Restores a network from a binary model file. This also automatically allocates the memory and topology for the network.

## Demos

These are found in the `demos` folder. Navigate to it and just run the makefile, then run them in order:
- `demo0.c`: Train an XOR model and store it in the `models` folder.
- `demo1.c`: Load the XOR model from the `models` folder and test it in real time.
- `demo2.c`: Train a simple spiral separation model.
- `plot_demo2.c`: Plots the output model of `demo2.c`. This will output a file named `output.ppm` that you'll need a `.ppm` viewer to view.

## Issues

- Doesn't work with OpenMP for some reason: I tried to parallelise it using `omp.h`, then it just didn't work anymore. It's probably because of race conditions and such.
- Doesn't output to a more universal format, just unreadable binary files: I really don't know of a universal multilayer perceptron file format.
- Because of the reason above, it's kinda hard to test it.
- Because it's single-header, it's kinda trying to do too many things. Too many here probably means two or three here but you do get the point.
- There should probably be new demos, the ones here have barely changed since the initial commit. But well, there aren't really new features anyway. But maybe I should write better demos, the ones here aren't compiled with `-Wall` for a reason, and it's not a good reason.
