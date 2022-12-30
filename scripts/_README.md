## 0. Note
Move the script files at `../src` where `main.py` exists. 
Run the script files at `../src`.

## 1. Train and test a model

### Train a model
Train a model, using `train_model.sh`.

### Test a model
Test a model's accuracy, using `test_model.sh`.

<!-- 
### Create example patches of a model
Create example patches of neurons in a model, using `example_patch.sh`.
-->

## 2. Embedding

### Neuron embedding of a base model
1. Create stimuli of neurons in the base model, using `stimulus.sh`.
2. Create neuron embedding of the base model, using `embedding_neuron.sh`.

### Image embedding from the base model
Create image embedding, using `embedding_img.sh`.

### (Approximated) neuron embedding of a non-base model
Create neuron embedding of a non-base model, using `embedding_proj.sh`.

### 2D embedding of the base model and non-base models
Create 2D embeddings of the base model and non-base models, using `embedding_2d.sh`.

## 3. Evaluate the importance of concept evolution in a model