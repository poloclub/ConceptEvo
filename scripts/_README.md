## Note
Move the script files at `../src` where `main.py` exists. 
Run the script files at `../src`.

## Train and test a model

### Train a model
Train a model, using `train_model.sh`.

### Test a model
Test a model's accuracy, using `test_model.sh`.

<!-- 
### Create example patches of a model
Create example patches of neurons in a model, using `example_patch.sh`.
-->

## Embedding

### Neuron embedding of a base model
1. Create stimuli of neurons in the base model, using `stimulus.sh`.
2. Create neuron embedding of the base model, using `neuron_embedding.sh`.

### Image embedding from the base model
Create image embedding, using `img_embedding.sh`.

### (Approximated) neuron embedding of a non-base model
Create neuron embedding of a non-base model, using `proj_embedding.sh`.

### 2D embedding of the base model and non-base models
Create 2D embeddings of the base model and non-base models, using `2d_embedding.sh`.

## Evaluate the importance of concept evolution in a model