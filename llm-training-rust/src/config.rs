use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub embedding_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub feed_forward_dim: usize,
    pub dropout_rate: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub checkpoint_interval: usize,
}

impl Config {
    pub fn from_file(file_path: &str) -> Self {
        let json_str = std::fs::read_to_string(file_path).expect("Failed to read config file");
        serde_json::from_str(&json_str).expect("Failed to parse config JSON")
    }
}