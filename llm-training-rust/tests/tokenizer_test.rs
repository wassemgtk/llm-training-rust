use llm_training_rust::tokenizer::Tokenizer;

#[test]
fn test_tokenizer_encode() {
    let vocab_file = "path/to/vocab.txt";
    let tokenizer = Tokenizer::new(vocab_file);

    let text = "This is a sample text.";
    let encoded = tokenizer.encode(text);

    assert!(!encoded.is_empty());
}

#[test]
fn test_tokenizer_decode() {
    let vocab_file = "path/to/vocab.txt";
    let tokenizer = Tokenizer::new(vocab_file);

    let text = "This is a sample text.";
    let encoded = tokenizer.encode(text);
    let decoded = tokenizer.decode(&encoded);

    assert_eq!(decoded, text);
}