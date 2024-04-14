use llm_training_rust::embedding::Embedding;
use llm_training_rust::config::Config;

#[test]
fn test_embedding_forward() {
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

    let embedding = Embedding::new(config.vocab_size, config.embedding_dim);

    let input = vec![1, 2, 3, 4];
    let output = embedding.forward(&input);

    assert_eq!(output.len(), input.len());
    assert_eq!(output[0].len(), config.embedding_dim);
}

#[test]
fn test_embedding_backward() {
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

    let mut embedding = Embedding::new(config.vocab_size, config.embedding_dim);

    let input = vec![1, 2, 3, 4];
    let output = embedding.forward(&input);

    let grad_output = vec![vec![1.0; config.embedding_dim]; input.len()];
    embedding.backward(&grad_output, &input);

    // Check if the gradients are computed for the embedding matrix
    // You can add more specific assertions based on your implementation
}