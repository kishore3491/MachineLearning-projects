# Image classification project - CIFAR 10

# Challenges
1. Small Dataset size: There is not much room for Train/Validation/Test split. Hence, Train data is shuffled, and used for both training and validation - 80/20. This resulted in very quick overfitting on Train data, and inflated accuracy scores just after a few epochs. Test data is untouched until evaluation.

# Pipeline:
1. FIFOQueue to provide dataset batches, read from binary format files with optional augmentation and shuffling.
2. A typical CNN with Max pooling, RELU activations and Dropouts.
3. Training pipeline which writes the variables to checkpoint after training.
4. Evaluation pipeline which reads variables from checkpoint and runs test data batches only once.

# Limitations:
1. Fixed size input images: The current model only accepts images of size=[32,32,3]. This is because of the need Flattening and Fully connected layers in the back, which expects a fixed size input. Publications like SPPNet suggest methods to accept variable size images.
