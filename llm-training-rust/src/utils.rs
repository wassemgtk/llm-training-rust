
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::model::Model;

pub fn read_lines(file_path: &str) -> Vec<String> {
    let file = File::open(file_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    reader.lines().map(|line| line.expect("Failed to read line")).collect()
}

pub fn write_lines(file_path: &str, lines: &[String]) {
    let mut file = File::create(file_path).expect("Failed to create file");
    for line in lines {
        writeln!(file, "{}", line).expect("Failed to write line");
    }
}

pub fn read_text_file(file_path: &str) -> String {
    std::fs::read_to_string(file_path).expect("Failed to read file")
}

pub fn write_text_file(file_path: &str, text: &str) {
    std::fs::write(file_path, text).expect("Failed to write file");
}

pub fn file_exists(file_path: &str) -> bool {
    Path::new(file_path).exists()
}

pub fn create_directory(dir_path: &str) {
    std::fs::create_dir_all(dir_path).expect("Failed to create directory");
}

pub fn save_model(model: &Model, file_path: &str) {
    let serialized_model = bincode::serialize(model).expect("Failed to serialize model");
    std::fs::write(file_path, serialized_model).expect("Failed to save model");
}

pub fn load_model(file_path: &str) -> Model {
    let serialized_model = std::fs::read(file_path).expect("Failed to read model file");
    bincode::deserialize(&serialized_model).expect("Failed to deserialize model")
}

#[derive(Serialize, Deserialize)]
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
    pub fn from_json(json_str: &str) -> Self {
        serde_json::from_str(json_str).expect("Failed to parse config JSON")
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Failed to serialize config to JSON")
    }

    pub fn save_to_file(&self, file_path: &str) {
        let json_str = self.to_json();
        write_text_file(file_path, &json_str);
    }

    pub fn load_from_file(file_path: &str) -> Self {
        let json_str = read_text_file(file_path);
        Self::from_json(&json_str)
    }
}