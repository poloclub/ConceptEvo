## 0. Preliminary Note: Script Setup and Execution
Before diving into the usage of these scripts, please ensure they are properly placed and executed:

- **❗️Move script files❗️**: Relocate all script files to the `../src` directory, where `main.py` is located. This ensures that the scripts can interact correctly with the main application. Run a script file within the `../src` directory for it to function as intended.

- **Details in scripts**: Each script includes comprehensive guidance on setting hyperparameters, as well as the location where the results will be saved.

- **Recommended Order**: Below is a suggested sequence for using these scripts. Feel free to skip any that are not applicable to your requirements.

## 1. Train and test a model
- Train a model
    - Train a model, using `train_model.sh`.

- Test a model
    - Test a model's accuracy, using `test_model.sh`.

- Create example patches 
    - Create example patches of neurons in a model, using `example_patch.sh`.

## 2. Embedding
The following steps are aligned with the steps described in Section 3.2 of the 📄[ConceptEvo paper](https://arxiv.org/abs/2203.16475).

### Step 1: Creating the base semantic space
This process involves creating a foundational semantic space, for a user-given base model.

- **Step 1.1: Finding stimuli**
    - **Description**: Generate stimuli for neurons in the base model.
    - **Script**: Use `stimulus.sh` to create these stimuli.

- **Step 1.2 and 1.3: Sampling co-activated neurons and learning neuron embedding**
    - **Description**: These steps involve sampling frequently co-activated neuron pairs and learning neuron embeddings based on these neuron pairs.
    - **Script**: Execute `neuron_embedding.sh`.

### Step 2: Unifying the semantic space of different models at different epochs

- **Step 2.1: Image embedding**
    - **Description**: This process involves randomly sampling training images and computing those images' embedding, in order to approximate neuron embeddings in the base model. 
    - **2.1.1 Randomly Sample Images**: 
        - **Objective**: Select a random subset of training images.
        - **Script**: Execute `sample_images.sh` to perform this action. 
    - **2.1.2 Compute Most Responsive Neurons**: 
        - **Objective**: Identify and focus on the neurons in the base model that are most responsive to each sampled image. 
        - **Details**: These responsive neurons are crucial in the gradient descent process for minimizing the difference between the original neuron embeddings and their approximations.
        - **Script**: Run `responsive_neurons.sh` for this analysis.
    - **2.1.3 Learn Image Embedding**: 
        - **Objective**: Learn embeddings of the sampled images, such that they help in reducing the gap between the original and approximated neuron embeddings from the base model.
        - **Script**: Use `image_embedding.sh` to execute this learning process. 
    - **2.1.4 Learn Image Embeddings that are not covered by the base models' neurons**
        - **Objectvie**: Indirectly represent images that are not covered by the base model's stimuli, focusing on common neuron activation by images.
        - **Script**: Use `indirect_image_embedding.sh`. 

- **Step 2.2: Approximating embedding of neurons in other models at different epochs**
    - **Description**: Project the embedding of neurons from non-base models onto the base semantic space using the learned image embeddings, thereby creating a unified semantic space across models and epochs.
    - **Scripts**:
        - First, run `proj_embedding.sh` for projection.
        - Then, execute `reduced_embedding.sh` to generate 2D embeddings of (either original or approximated) neuron embeddings of different models, using UMAP.

## 3. Evaluate the importance of concept evolution in a model



## 4. Generate example patches of neurons