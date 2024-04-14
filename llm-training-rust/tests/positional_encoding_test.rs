use llm_training_rust::positional_encoding::PositionalEncoding;
use llm_training_rust::config::Config;

#[test]
fn test_positional_encoding_forward() {
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

    let positional_encoding = PositionalEncoding::new(config.max_seq_len, config.embedding_dim);

    let seq_len = 10;
    let output = positional_encoding.forward(seq_len);

    assert_eq!(output.len(), seq_len);
    assert_eq!(output[0].len(), config.embedding_dim);
}