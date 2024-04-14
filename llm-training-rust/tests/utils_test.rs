use llm_training_rust::utils::{read_lines, write_lines, read_text_file, write_text_file, file_exists, create_directory, save_model, load_model, Config};
use llm_training_rust::model::Model;

#[test]
fn test_read_write_lines() {
    let file_path = "path/to/file.txt";
    let lines = vec!["Line 1".to_string(), "Line 2".to_string(), "Line 3".to_string()];

    write_lines(file_path, &lines);
    let read_lines = read_lines(file_path);

    assert_eq!(read_lines, lines);
}

#[test]
fn test_read_write_text_file() {
    let file_path = "path/to/file.txt";
    let text = "This is a sample text.";

    write_text_file(file_path, text);
    let read_text = read_text_file(file_path);

    assert_eq!(read_text, text);
}

#[test]
fn test_file_exists() {
    let file_path = "path/to/file.txt";

    assert!(file_exists(file_path));
}

#[test]
fn test_create_directory() {
    let dir_path = "path/to/directory";

    create_directory(dir_path);

    assert!(std::path::Path::new(dir_path).exists());
}

#[test]
fn test_save_load_model() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 1,
        checkpoint_interval: 10,
    };

    let model = Model::new(&config);
    let file_path = "path/to/model.bin";

    save_model(&model, file_path);
    let loaded_model = load_model(file_path);

    // Compare the loaded model with the original model
    // You can add more specific assertions based on your implementation
}

#[test]
fn test_config_from_to_json() {
    let json_str = r#"
        {
            "vocab_size": 100,
            "max_seq_len": 20,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "feed_forward_dim": 64,
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "batch_size": 2,
            "num_epochs": 1,
            "checkpoint_interval": 10
        }
    "#;

    let config = Config::from_json(json_str);
    let serialized_json = config.to_json();

    assert_eq!(serialized_json, json_str.trim());
}

#[test]
fn test_config_save_load_file() {
    let config = Config {
        vocab_size: 100,
        max_seq_len: 20,
        embedding_dim: 32,
        num_layers: 2,
        num_heads: 4,
        feed_forward_dim: 64,
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 1,
        checkpoint_interval: 10,
    };

    let file_path = "path/to/config.json";

    config.save_to_file(file_path);
    let loaded_config = Config::load_from_file(file_path);

    assert_eq!(config, loaded_config);
}