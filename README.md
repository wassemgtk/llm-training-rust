# Language Model Training in Rust

This project is an implementation of a Language Model (LLM) training framework in Rust. It provides a set of modules and utilities for building, training, and evaluating language models using the transformer architecture.

## Features

- Transformer-based language model architecture
- Multi-head self-attention mechanism
- Positional encoding for sequence information
- Feed-forward neural network layers
- Embedding layer for input tokens
- Layer normalization for stable training
- GELU activation function
- Adam optimizer for parameter updates
- Data loading and batching utilities
- Tokenization and vocabulary handling
- Model checkpointing and loading
- Comprehensive test suite for all modules

## Project Structure

The project follows a standard Rust project structure:

```
llm-training-rust/
  ├── src/
  │   ├── main.rs
  │   ├── config.rs
  │   ├── model.rs
  │   ├── attention.rs
  │   ├── layer_norm.rs
  │   ├── gelu.rs
  │   ├── embedding.rs
  │   ├── positional_encoding.rs
  │   ├── feed_forward.rs
  │   ├── transformer.rs
  │   ├── optimizer.rs
  │   ├── data_loader.rs
  │   ├── tokenizer.rs
  │   └── utils.rs
  ├── tests/
  │   ├── model_test.rs
  │   ├── attention_test.rs
  │   ├── layer_norm_test.rs
  │   ├── gelu_test.rs
  │   ├── embedding_test.rs
  │   ├── positional_encoding_test.rs
  │   ├── feed_forward_test.rs
  │   ├── transformer_test.rs
  │   ├── optimizer_test.rs
  │   ├── data_loader_test.rs
  │   ├── tokenizer_test.rs
  │   └── utils_test.rs
  ├── data/
  │   ├── tiny_shakespeare_train.txt
  │   └── tiny_shakespeare_val.txt
  ├── Cargo.toml
  └── README.md
```

- `src/`: Contains the main source code files for the language model implementation.
- `tests/`: Contains the unit tests for each module.
- `data/`: Contains the training and validation data files.

## Getting Started

### Prerequisites

- Rust (stable version)
- Cargo (Rust package manager)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/wassemgtk/llm-training-rust.git
   ```

2. Navigate to the project directory:
   ```
   cd llm-training-rust
   ```

### Training

To train the language model, follow these steps:

1. Prepare your training data:
   - Place your training data file (e.g., `tiny_shakespeare_train.txt`) in the `data/` directory.
   - Update the `file_path` variable in `main.rs` to point to your training data file.

2. Configure the model hyperparameters in the `Config` struct in `config.rs`.

3. Run the training script:
   ```
   cargo run --release
   ```

4. Monitor the training progress and metrics logged to the console.

### Testing

To run the test suite and ensure the correctness of the implemented modules, use the following command:

```
cargo test
```

This will execute all the test functions in the `tests/` directory.

### Generating Text

To generate text using a trained model checkpoint, follow these steps:

1. Make sure you have a trained model checkpoint file (e.g., `model_checkpoint.bin`) in the project directory.

2. Update the `model_checkpoint` variable in `main.rs` to point to your trained model checkpoint file.

3. Set the desired generation parameters (e.g., `max_new_tokens`, `temperature`) in the `generate` function call in `main.rs`.

4. Run the text generation script:
   ```
   cargo run --release --bin generate
   ```

5. The generated text will be printed to the console.

## Configuration

The `Config` struct in `config.rs` contains the hyperparameters and configuration settings for the language model. You can modify these values to experiment with different model architectures and training setups.

## Model Checkpointing

During training, the model checkpoints will be saved in the project directory with the specified checkpoint interval. You can use these checkpoints to resume training from a previous state or to generate text using a trained model.

## Dependencies

The project relies on the following dependencies:

- `rand`: Random number generation for sampling and initialization.
- `serde`: Serialization and deserialization of data structures.
- `serde_json`: JSON serialization and deserialization.
- `bincode`: Binary serialization and deserialization.
- `approx` (dev): Approximate floating-point comparisons for testing.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The transformer architecture is based on the paper "Attention Is All You Need" by Vaswani et al.
- The implementation draws inspiration from various open-source language model implementations in the Rust ecosystem.

