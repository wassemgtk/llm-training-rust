use std::ops::{AddAssign, SubAssign};

pub struct LayerNorm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        let gamma = vec![1.0; dim];
        let beta = vec![0.0; dim];
        let eps = 1e-5;

        Self { gamma, beta, eps }
    }

    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut output = Vec::with_capacity(input.len());
        let dim = input[0].len();

        for i in 0..input.len() {
            let mean = input[i].iter().sum::<f64>() / dim as f64;
            let variance = input[i].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
            let std_dev = (variance + self.eps).sqrt();

            let normalized = input[i]
                .iter()
                .zip(&self.gamma)
                .zip(&self.beta)
                .map(|((&x, &g), &b)| (x - mean) / std_dev * g + b)
                .collect();

            output.push(normalized);
        }

        output
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>], input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut grad_input = Vec::with_capacity(input.len());
        let dim = input[0].len();

        for i in 0..input.len() {
            let mean = input[i].iter().sum::<f64>() / dim as f64;
            let variance = input[i].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
            let std_dev = (variance + self.eps).sqrt();

            let mut d_gamma = vec![0.0; dim];
            let mut d_beta = vec![0.0; dim];
            let mut d_input = vec![0.0; dim];

            for j in 0..dim {
                let normalized = (input[i][j] - mean) / std_dev;
                d_gamma[j] = grad_output[i][j] * normalized;
                d_beta[j] = grad_output[i][j];
            }

            let mut d_norm = vec![0.0; dim];
            for j in 0..dim {
                d_norm[j] = grad_output[i][j] * self.gamma[j];
            }

            let d_variance = d_norm
                .iter()
                .zip(&input[i])
                .map(|(&d, &x)| d * (x - mean) * -0.5 * (variance + self.eps).powf(-1.5))
                .sum::<f64>();

            let d_mean = d_norm.iter().sum::<f64>() * -1.0 / std_dev
                + d_variance * input[i].iter().map(|&x| -2.0 * (x - mean)).sum::<f64>() / dim as f64;

            for j in 0..dim {
                d_input[j] = d_norm[j] / std_dev + d_variance * 2.0 * (input[i][j] - mean) / dim as f64 + d_mean / dim as f64;
            }

            self.gamma.iter_mut().zip(&d_gamma).for_each(|(g, d)| g.add_assign(d));
            self.beta.iter_mut().zip(&d_beta).for_each(|(b, d)| b.add_assign(d));

            grad_input.push(d_input);
        }

        grad_input
    }
}