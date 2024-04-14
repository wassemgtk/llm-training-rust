use crate::config::Config;
use crate::model::Model;
use crate::data_loader::DataLoader;
use crate::tokenizer::Tokenizer;

mod config;
mod model;
mod attention;
mod layer_norm;
mod gelu;
mod embedding;
mod positional_encoding;
mod feed_forward;
mod transformer;
mod optimizer;
mod data_loader;
mod tokenizer;
mod utils;

fn main() {
    // Load the configuration
    let config = Config::from_file("config.json");

    // Initialize the tokenizer
    let tokenizer = Tokenizer::new("vocab.txt");

    // Load the training data
    let train_data = DataLoader::new("data/tiny_shakespeare_train.txt", config.batch_size, config.seq_len, &tokenizer);

    // Initialize the model
    let mut model = Model::new(&config);

    // Train the model
    model.train(&train_data, &config);

    // Generate text
    let prompt = "To be, or not to be";
    let generated_text = model.generate(prompt, &tokenizer, &config);

    println!("Generated text: {}", generated_text);
}