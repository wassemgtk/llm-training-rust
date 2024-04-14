use std::collections::HashMap;

pub struct Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    pub eos_id: usize,
    pub pad_id: usize,
}

impl Tokenizer {
    pub fn new(vocab_file: &str) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        let vocab = std::fs::read_to_string(vocab_file).expect("Failed to read vocabulary file");
        for (id, token) in vocab.lines().enumerate() {
            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        let eos_id = token_to_id.len();
        let pad_id = token_to_id.len() + 1;

        token_to_id.insert("<eos>".to_string(), eos_id);
        id_to_token.insert(eos_id, "<eos>".to_string());
        token_to_id.insert("<pad>".to_string(), pad_id);
        id_to_token.insert(pad_id, "<pad>".to_string());

        Self {
            token_to_id,
            id_to_token,
            eos_id,
            pad_id,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|token| self.token_to_id.get(token).unwrap_or(&self.eos_id))
            .cloned()
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.id_to_token.get(&id).unwrap_or(&"<unk>".to_string()))
            .cloned()
            .collect::<Vec<String>>()
            .join(" ")
    }
}