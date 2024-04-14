use llm_training_rust::optimizer::AdamOptimizer;
use llm_training_rust::model::Model;
use llm_training_rust::config::Config;

#[test]
fn test_optimizer_step() {
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
    let mut optimizer = AdamOptimizer::new(config.learning_rate);

    // Perform a forward pass and backward pass to compute gradients
    let input = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let target = vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9]];
    let (logits, _) = model.forward(&input, Some(&target));
    model.backward(&logits, &input, &target);

    // Perform an optimizer step
    optimizer.step(&mut model);

    // Check if the model parameters are updated
    // You can add more specific assertions based on your implementation
}