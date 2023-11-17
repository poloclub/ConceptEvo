## 0. Preliminary Note: Script Setup and Execution
Before diving into the usage of these scripts, please ensure they are properly placed and executed:

- **Move Script Files**: Relocate all script files to the `../src` directory, where `main.py` is located. This ensures that the scripts can interact correctly with the main application.

- **Execute Scripts in `../src`**: Run a script file within the `../src` directory for it to function as intended.

- **Recommended Order**: Below is a suggested sequence for using these scripts. Feel free to skip any that are not applicable to your requirements.

## 1. Train and test a model
- Train a model
    - Train a model, using `train_model.sh`.

- Test a model
    - Test a model's accuracy, using `test_model.sh`.

- Create example patches 
    - - Create example patches of neurons in a model, using `example_patch.sh`.

## 2. Embedding

- Neuron embedding of a base model
    1. Create stimuli of neurons in the base model, using `stimulus.sh`.
    2. Create neuron embedding of the base model, using `embedding_neuron.sh`.

- Image embedding from the base model
    - Create image embedding, using `embedding_img.sh`.

- (Approximated) neuron embedding of a non-base model
    - Create neuron embedding of a non-base model, using `embedding_proj.sh`.

- 2D embedding of the base model and non-base models
    - Create 2D embeddings of the base model and non-base models, using `embedding_2d.sh`.

## 3. Evaluate the importance of concept evolution in a model