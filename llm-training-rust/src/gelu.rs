pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((x / (2.0_f64.sqrt())).tanh()))
}

pub fn gelu_backward(grad_output: f64, input: f64) -> f64 {
    let sqrt_2_over_pi = 0.7978845608028654;
    let cdf = 0.5 * (1.0 + ((input * sqrt_2_over_pi).tanh()));
    let pdf = (-0.5 * input.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();
    grad_output * (cdf + input * pdf)
}