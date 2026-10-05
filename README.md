# Very simple machine learning library in C

A small single-header-only multilayer perceptron 'library' written in C (`mlp.h`).

## Enums
- **Activation functions**: `Activation`
  - `ACT_linear`: Linear.
  - `ACT_relu`: ReLU.
  - `ACT_leaky_relu`: Leaky ReLU.
  - `ACT_logistic`: Logistic / Sigmoid.
  - `ACT_tanh`: Tanh.
- **Weight Initializations**: `Initialisation`
  - `INIT_zero`: Zero Initialization.
  - `INIT_Xavier`: Xavier Initialization.
  - `INIT_He`: He Initialization
- **Loss Functions**: `Loss_func`
  - `LOSS_MSE`: Mean Squared Error.
  - `LOSS_ASE`: Absolute Squared Error 
  - `LOSS_binary_cross_entropy`: Binary Cross-Entropy. (I should probably change the name)
- **Error Codes**: `MLPE_CODE`
  - `MLPE_SUCCESS` (value 0): Success.
  - `MLPE_NULL_PTR`: Null pointer passed into function;
  - `MLPE_MALLOC`: Memory allocation failed.
  - `MLPE_FILE_ERROR`: Errors related to file opening, reading, or writing.
  - `MLPE_WHAT`: Unknown error (reserved).

## API Reference
### Topology Management
- `MLPE_CODE Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons)`: Initializes the network structure and caches weight/value offsets.
- `MLPE_CODE Topology_free(Topology* topo)`: Frees topology memory.

### Network Lifecycle
- `MLPE_CODE MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act)`: Allocates memory using a single contiguous memory pool for weights, biases, values, and gradient deltas.
- `MLPE_CODE MLP_populate(MLP* input, Initialisation initialisation_method, unsigned int seed)`: Populates weights using one of the initialisations.
- `MLPE_CODE MLP_free(MLP* input)`: Frees all memory associated with the network pool and its topology.

### Execution & Training
- `MLPE_CODE MLP_set_inputs(MLP* mlp, float values[])`: Loads input features into the input layer.
- `MLPE_CODE MLP_evaluate(MLP* mlp)`: Performs forward propagation.
- `MLPE_CODE MLP_get_outputs(MLP* mlp, float* rop)`: Extracts predictions from the output layer.
- `MLPE_CODE MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func)`: Computes gradients via backpropagation for a single training sample.
- `MLPE_CODE MLP_update_weights(MLP* mlp, float learning_rate)`: Applies gradient descent to update weights and biases.
- `MLPE_CODE MLP_train_step(MLP* mlp, float inputs[], float targets[], float learning_rate, Loss_func loss_func)`: Runs a single forward pass, backpropagation, and weight update.
- `MLPE_CODE MLP_train([a lot of inputs, see the .h file])`: Standard batch training loop across multiple epochs.
- `MLPE_CODE MLP_train_logged([a lot of inputs, see the .h file])`: Training loop with periodic loss evaluation and logging callback support.

### Saving & Loading
- `MLPE_CODE MLP_save(const MLP* mlp, const char* filename)`: Saves network a network to memory.
- `MLPE_CODE MLP_load(MLP* mlp, const char* filename)`: Restores a network from a binary model file. This also automatically allocates the memory and topology for the network.

## Compilation

Compile it however you want! It's I made it so it depends on very little things. (use `-lm`)

## Demos
These are found in the `demos` folder. Navigate to it and just run the makefile, then run them in order:
- `demo0.c`: Train an XOR model and store it in the `models` folder.
- `demo1.c`: Load the XOR model from the `models` folder and test it in real time.
- `demo2.c`: Train a simple spiral separation model.
- `plot_demo2.c`: Plots the output model of `demo2.c`. This will output a file named `output.ppm` that you'll need a `.ppm` viewer to view.

## Issues
- Doesn't work with OpenMP for some reason: I tried to parallelise it using `omp.h`, then it just didn't work anymore. It's probably because of race conditions and such.
- Doesn't output to a more universal format, just unreadable binary files: I really don't know of a universal MLP format.
- Because of the last reason, it's kinda hard to test it.
- The training functions' signatures are massive!
