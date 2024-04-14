use crate::config::Config;
use crate::transformer::Transformer;
use crate::embedding::Embedding;
use crate::positional_encoding::PositionalEncoding;
use crate::layer_norm::LayerNorm;
use crate::optimizer::AdamOptimizer;
use crate::data_loader::DataLoader;
use crate::tokenizer::Tokenizer;

pub struct Model {
    embedding: Embedding,
    positional_encoding: PositionalEncoding,
    transformer: Transformer,
    layer_norm: LayerNorm,
    linear: Linear,
}

impl Model {
    pub fn new(config: &Config) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.embedding_dim);
        let positional_encoding = PositionalEncoding::new(config.max_seq_len, config.embedding_dim);
        let transformer = Transformer::new(config);
        let layer_norm = LayerNorm::new(config.embedding_dim);
        let linear = Linear::new(config.embedding_dim, config.vocab_size);

        Self {
            embedding,
            positional_encoding,
            transformer,
            layer_norm,
            linear,
        }
    }

    pub fn forward(&self, input: &[usize], target: Option<&[usize]>) -> (Vec<Vec<f64>>, Option<f64>) {
        let mut embeddings = self.embedding.forward(input);
        let positional_encodings = self.positional_encoding.forward(input.len());

        for (embedding, positional_encoding) in embeddings.iter_mut().zip(positional_encodings.iter()) {
            for (e, p) in embedding.iter_mut().zip(positional_encoding.iter()) {
                *e += p;
            }
        }

        let transformer_output = self.transformer.forward(&embeddings);
        let normed_output = self.layer_norm.forward(&transformer_output);
        let logits = self.linear.forward(&normed_output);

        let loss = match target {
            Some(target) => {
                let loss = self.cross_entropy_loss(&logits, target);
                Some(loss)
            }
            None => None,
        };

        (logits, loss)
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>], input: &[usize], target: &[usize]) {
        let grad_linear = self.linear.backward(grad_output);
        let grad_layer_norm = self.layer_norm.backward(&grad_linear, &self.transformer.output);
        let grad_transformer = self.transformer.backward(&grad_layer_norm);

        let mut grad_embeddings = grad_transformer;
        let grad_positional_encodings = self.positional_encoding.backward(&grad_embeddings);

        for (grad_embedding, grad_positional_encoding) in grad_embeddings.iter_mut().zip(grad_positional_encodings.iter()) {
            for (ge, gp) in grad_embedding.iter_mut().zip(grad_positional_encoding.iter()) {
                *ge += gp;
            }
        }

        self.embedding.backward(&grad_embeddings, input);
    }

    pub fn train(&mut self, data_loader: &DataLoader, config: &Config) {
        let mut optimizer = AdamOptimizer::new(config.learning_rate);

        for epoch in 0..config.num_epochs {
            let mut total_loss = 0.0;

            for (batch_input, batch_target) in data_loader.iter() {
                let (logits, loss) = self.forward(&batch_input, Some(&batch_target));
                self.backward(&logits, &batch_input, &batch_target);

                optimizer.step(self);
                total_loss += loss.unwrap();
            }

            let avg_loss = total_loss / data_loader.len() as f64;
            println!("Epoch: {}, Loss: {}", epoch + 1, avg_loss);

            if (epoch + 1) % config.checkpoint_interval == 0 {
                self.save_checkpoint(&format!("checkpoint_epoch_{}.pt", epoch + 1));
            }
        }
    }

    pub fn generate(&self, prompt: &str, tokenizer: &Tokenizer, config: &Config) -> String {
        let mut input_ids = tokenizer.encode(prompt);
        let mut generated_ids = Vec::new();

        for _ in 0..config.max_seq_len {
            let (logits, _) = self.forward(&input_ids, None);
            let last_logits = logits.last().unwrap();
            let probs = softmax(last_logits);
            let next_id = sample_multinomial(&probs);

            if next_id == tokenizer.eos_id {
                break;
            }

            generated_ids.push(next_id);
            input_ids.push(next_id);
            input_ids = input_ids.split_off(1);
        }

        let generated_text = tokenizer.decode(&generated_ids);
        generated_text
    }

    fn cross_entropy_loss(&self, logits: &[Vec<f64>], target: &[usize]) -> f64 {
        let mut loss = 0.0;

        for (logit, &target_id) in logits.iter().zip(target.iter()) {
            let probs = softmax(logit);
            loss -= probs[target_id].ln();
        }

        loss / logits.len() as f64
    }

    fn save_checkpoint(&self, path: &str) {
        // Implement saving the model checkpoint to a file
        // You can use serde and bincode to serialize the model
    }

    fn load_checkpoint(path: &str) -> Self {
        // Implement loading the model checkpoint from a file
        // You can use serde and bincode to deserialize the model
        unimplemented!()
    }
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum_exp_x = exp_x.iter().sum::<f64>();
    exp_x.into_iter().map(|exp_xi| exp_xi / sum_exp_x).collect()
}

fn sample_multinomial(probs: &[f64]) -> usize {
    let mut rng = rand::thread_rng();
    let mut cum_probs = probs.to_vec();
    for i in 1..cum_probs.len() {
        cum_probs[i] += cum_probs[i - 1];
    }
    let r: f64 = rng.gen();
    cum_probs.iter().position(|&p| p > r).unwrap()
}