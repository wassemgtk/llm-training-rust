use crate::config::Config;
use crate::attention::Attention;
use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;

pub struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(config: &Config) -> Self {
        let attention = Attention::new(config);
        let feed_forward = FeedForward::new(config);
        let layer_norm1 = LayerNorm::new(config.embedding_dim);
        let layer_norm2 = LayerNorm::new(config.embedding_dim);

        Self {
            attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
        }
    }

    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let attention_output = self.attention.forward(input);
        let residual1 = input.iter().zip(attention_output.iter()).map(|(x, y)| {
            x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect()
        }).collect::<Vec<Vec<f64>>>();
        let norm1 = self.layer_norm1.forward(&residual1);

        let feed_forward_output = self.feed_forward.forward(&norm1);
        let residual2 = norm1.iter().zip(feed_forward_output.iter()).map(|(x, y)| {
            x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect()
        }).collect::<Vec<Vec<f64>>>();
        let norm2 = self.layer_norm2.forward(&residual2);

        norm2
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let grad_norm2 = self.layer_norm2.backward(grad_output, &self.feed_forward.output);
        let (grad_norm1, grad_feed_forward_output) = grad_norm2.iter().zip(self.feed_forward.output.iter())
            .map(|(x, y)| (x.clone(), x.clone()))
            .unzip();

        let grad_feed_forward = self.feed_forward.backward(&grad_feed_forward_output);
        let grad_residual1 = self.layer_norm1.backward(&grad_norm1, &self.attention.output);
        let (grad_input, grad_attention_output) = grad_residual1.iter().zip(self.attention.output.iter())
            .map(|(x, y)| (x.clone(), x.clone()))
            .unzip();

        let grad_attention = self.attention.backward(&grad_attention_output);

        grad_input.iter().zip(grad_attention.iter())
            .map(|(x, y)| x.iter().zip(y.iter()).map(|(&a, &b)| a + b).collect())
            .collect()
    }
}

pub struct Transformer {
    layers: Vec<TransformerLayer>,
    pub output: Vec<Vec<f64>>,
}

impl Transformer {
    pub fn new(config: &Config) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| TransformerLayer::new(config))
            .collect();
        let output = Vec::new();

        Self { layers, output }
    }

    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut output = input.to_vec();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        self.output = output.clone();
        output
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut grad_input = grad_output.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad_input = layer.backward(&grad_input);
        }
        grad_input
    }
}