use crate::config::Config;
use crate::linear::Linear;
use crate::gelu::gelu;

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(config: &Config) -> Self {
        let linear1 = Linear::new(config.embedding_dim, config.feed_forward_dim);
        let linear2 = Linear::new(config.feed_forward_dim, config.embedding_dim);
        let dropout = Dropout::new(config.dropout_rate);
        Self {
            linear1,
            linear2,
            dropout,
        }
    }

    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut hidden = self.linear1.forward(input);
        hidden = hidden.iter().map(|h| h.iter().map(|&x| gelu(x)).collect()).collect();
        let output = self.linear2.forward(&hidden);
        self.dropout.forward(&output)
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let grad_dropout = self.dropout.backward(grad_output);
        let grad_linear2 = self.linear2.backward(&grad_dropout);
        let grad_gelu = grad_linear2
            .iter()
            .zip(self.linear1.output.iter())
            .map(|(go, o)| go.iter().zip(o.iter()).map(|(&go_i, &o_i)| go_i * (1.0 - o_i.tanh().powi(2))).collect())
            .collect();
        self.linear1.backward(&grad_gelu)
    }
}