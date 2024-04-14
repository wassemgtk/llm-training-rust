use std::f64::consts::PI;

pub struct PositionalEncoding {
    pub encodings: Vec<Vec<f64>>,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
        let mut encodings = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let mut encoding = Vec::with_capacity(embedding_dim);
            for i in 0..embedding_dim {
                let angle = pos as f64 / (10000.0_f64).powf((2 * i) as f64 / embedding_dim as f64);
                encoding.push(angle.sin());
                encoding.push(angle.cos());
            }
            encodings.push(encoding);
        }
        Self { encodings }
    }

    pub fn forward(&self, seq_len: usize) -> Vec<Vec<f64>> {
        self.encodings[..seq_len].to_vec()
    }
}