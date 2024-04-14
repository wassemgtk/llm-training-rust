use llm_training_rust::transformer::{TransformerLayer, Transformer};
use llm_training_rust::config::Config;

#[test]
fn test_transformer_layer_forward() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 1,
        checkpoint_interval: 10,
    };

    let mut transformer_layer = TransformerLayer::new(&config);

    let input = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let output = transformer_layer.forward(&input);

    assert_eq!(output.len(), config.max_seq_len);
    assert_eq!(output[0].len(), config.embedding_dim);
}

#[test]
fn test_transformer_layer_backward() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 1,
        checkpoint_interval: 10,
    };

    let mut transformer_layer = TransformerLayer::new(&config);

    let input = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let output = transformer_layer.forward(&input);

    let grad_output = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let grad_input = transformer_layer.backward(&grad_output);

    assert_eq!(grad_input.len(), config.max_seq_len);
    assert_eq!(grad_input[0].len(), config.embedding_dim);
}

#[test]
fn test_transformer_forward() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 1,
        checkpoint_interval: 10,
    };

    let mut transformer = Transformer::new(&config);

    let input = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let output = transformer.forward(&input);

    assert_eq!(output.len(), config.max_seq_len);
    assert_eq!(output[0].len(), config.embedding_dim);
}

#[test]
fn test_transformer_backward() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_