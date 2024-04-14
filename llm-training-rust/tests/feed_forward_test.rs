use llm_training_rust::feed_forward::FeedForward;
use llm_training_rust::config::Config;

#[test]
fn test_feed_forward_forward() {
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

    let mut feed_forward = FeedForward::new(&config);

    let input = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let output = feed_forward.forward(&input);

    assert_eq!(output.len(), config.max_seq_len);
    assert_eq!(output[0].len(), config.embedding_dim);
}

#[test]
fn test_feed_forward_backward() {
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

    let mut feed_forward = FeedForward::new(&config);

    let input = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let output = feed_forward.forward(&input);

    let grad_output = vec![vec![1.0; config.embedding_dim]; config.max_seq_len];
    let grad_input = feed_forward.backward(&grad_output);

    assert_eq!(grad_input.len(), config.max_seq_len);
    assert_eq!(grad_input[0].len(), config.embedding_dim);
}