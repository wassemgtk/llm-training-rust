use llm_training_rust::model::Model;
use llm_training_rust::config::Config;

#[test]
fn test_model_forward() {
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

    let mut model = Model::new(&config);

    let input = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let target = vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9]];

    let (logits, loss) = model.forward(&input, Some(&target));

    assert_eq!(logits.len(), 2);
    assert_eq!(logits[0].len(), 4);
    assert_eq!(logits[0][0].len(), config.vocab_size);

    assert!(loss.is_some());
}

#[test]
fn test_model_backward() {
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

    let mut model = Model::new(&config);

    let input = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let target = vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9]];

    let (logits, _) = model.forward(&input, Some(&target));
    model.backward(&logits, &input, &target);

    // Check if the gradients are computed for all parameters
    // You can add more specific assertions based on your implementation
}