use llm_training_rust::layer_norm::LayerNorm;
use approx::assert_abs_diff_eq;

#[test]
fn test_layer_norm_forward() {
    let dim = 32;
    let mut layer_norm = LayerNorm::new(dim);

    let input = vec![vec![1.0; dim]; 3];
    let output = layer_norm.forward(&input);

    assert_eq!(output.len(), input.len());
    assert_eq!(output[0].len(), dim);
}

#[test]
fn test_layer_norm_backward() {
    let dim = 32;
    let mut layer_norm = LayerNorm::new(dim);

    let input = vec![vec![1.0; dim]; 3];
    let output = layer_norm.forward(&input);

    let grad_output = vec![vec![1.0; dim]; 3];
    let grad_input = layer_norm.backward(&grad_output, &input);

    assert_eq!(grad_input.len(), input.len());
    assert_eq!(grad_input[0].len(), dim);
}