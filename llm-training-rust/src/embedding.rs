pub struct Embedding {
    embedding_matrix: Vec<Vec<f64>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut embedding_matrix = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let embedding = Self::truncated_normal(embedding_dim);
            embedding_matrix.push(embedding);
        }

        Self { embedding_matrix }
    }

    pub fn forward(&self, input: &[usize]) -> Vec<Vec<f64>> {
        input.iter().map(|&idx| self.embedding_matrix[idx].clone()).collect()
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>], input: &[usize]) {
        for (idx, grad) in input.iter().zip(grad_output) {
            for (embedding_val, grad_val) in self.embedding_matrix[*idx].iter_mut().zip(grad) {
                *embedding_val += grad_val;
            }
        }
    }

    fn truncated_normal(dim: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let normal_dist = rand::distributions::Normal::new(0.0, 1.0);
        let mut embedding = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut val = normal_dist.sample(&mut rng);
            while val < -2.0 || val > 2.0 {
                val = normal_dist.sample(&mut rng);
            }
            embedding.push(val);
        }
        embedding
    }
}