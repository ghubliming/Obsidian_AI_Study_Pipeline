# Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks. They form the foundation of deep learning, which has revolutionized fields like computer vision, natural language processing, and speech recognition.

## Basic Structure

### Neurons (Nodes)
A neuron is the basic unit of a neural network. It receives inputs, applies a transformation, and produces an output.

The mathematical representation of a neuron:
$$y = f(\sum_{i=1}^{n} w_i x_i + b)$$

Where:
- $x_i$ are the inputs
- $w_i$ are the weights
- $b$ is the bias
- $f$ is the activation function

### Layers
Neural networks are organized in layers:
- **Input Layer**: Receives the initial data
- **Hidden Layer(s)**: Process the data through weighted connections
- **Output Layer**: Produces the final result

## Activation Functions

### Common Activation Functions

1. **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
   - Range: (0, 1)
   - Problem: Vanishing gradient

2. **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$
   - Most popular in deep learning
   - Solves vanishing gradient problem

3. **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
   - Range: (-1, 1)
   - Zero-centered output

4. **Softmax**: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$
   - Used in multi-class classification
   - Outputs probability distribution

## Training Process

### Forward Propagation
Data flows from input to output layer, with each layer transforming the input using weights, biases, and activation functions.

### Backpropagation
The process of updating weights by propagating the error backward through the network.

1. Calculate loss function
2. Compute gradients using chain rule
3. Update weights using gradient descent

### Loss Functions

- **Mean Squared Error (MSE)**: For regression
  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Cross-Entropy**: For classification
  $$H(p, q) = -\sum_{i} p_i \log(q_i)$$

## Deep Learning Architectures

### Convolutional Neural Networks (CNNs)
Specialized for processing grid-like data such as images.

Key components:
- **Convolutional Layers**: Apply filters to detect features
- **Pooling Layers**: Reduce spatial dimensions
- **Fully Connected Layers**: Traditional neural network layers

### Recurrent Neural Networks (RNNs)
Designed for sequential data like time series or text.

Types:
- **Vanilla RNN**: Basic recurrent structure
- **LSTM (Long Short-Term Memory)**: Handles long-term dependencies
- **GRU (Gated Recurrent Unit)**: Simplified version of LSTM

### Transformers
Modern architecture that uses self-attention mechanisms.
- Used in large language models (GPT, BERT)
- Highly parallelizable
- Excellent for sequence-to-sequence tasks

## Optimization Techniques

### Gradient Descent Variants
- **Batch Gradient Descent**: Uses entire dataset
- **Stochastic Gradient Descent (SGD)**: Uses single sample
- **Mini-batch Gradient Descent**: Uses small batches

### Advanced Optimizers
- **Adam**: Adaptive learning rates
- **RMSprop**: Divides learning rate by running average of gradients
- **AdaGrad**: Adapts learning rate based on historical gradients

## Regularization Techniques

### Dropout
Randomly sets a fraction of input units to 0 during training to prevent overfitting.

### Batch Normalization
Normalizes inputs to each layer, leading to faster training and better performance.

### Weight Decay (L2 Regularization)
Adds penalty term to loss function:
$$L_{total} = L_{original} + \lambda \sum w_i^2$$

## Applications

1. **Computer Vision**
   - Image classification
   - Object detection
   - Image segmentation

2. **Natural Language Processing**
   - Language translation
   - Sentiment analysis
   - Text generation

3. **Speech Recognition**
   - Voice assistants
   - Transcription services

4. **Recommendation Systems**
   - Product recommendations
   - Content filtering

## Challenges and Considerations

- **Computational Requirements**: Deep networks need significant computing power
- **Data Requirements**: Need large amounts of labeled data
- **Interpretability**: "Black box" nature makes it hard to understand decisions
- **Overfitting**: Complex models can memorize training data

## Related Notes
- [[Machine Learning Fundamentals]] - Basic ML concepts
- [[Python for Data Science]] - Implementation tools

#deep-learning #neural-networks #ai #machine-learning #mathematics