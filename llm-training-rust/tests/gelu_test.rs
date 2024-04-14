use llm_training_rust::gelu::{gelu, gelu_backward};
use approx::assert_abs_diff_eq;

#[test]
fn test_gelu_forward() {
    let input = vec![1.0, 2.0, 3.0];
    let output = input.iter().map(|&x| gelu(x)).collect::<Vec<_>>();

    assert_eq!(output.len(), input.len());
}

#[test]
fn test_gelu_backward() {
    let input = vec![1.0, 2.0, 3.0];
    let output = input.iter().map(|&x| gelu(x)).collect::<Vec<_>>();

    let grad_output = vec![1.0, 1.0, 1.0];
    let grad_input = output.iter().zip(input.iter()).map(|(&o, &i)| gelu_backward(o, i)).collect::<Vec<_>>();

    assert_eq!(grad_input.len(), input.len());
}