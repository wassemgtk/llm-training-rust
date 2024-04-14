use crate::config::Config;
use crate::linear::Linear;
use crate::dropout::Dropout;

pub struct Attention {
    query_matrix: Linear,
    key_matrix: Linear,
    value_matrix: Linear,
    output_matrix: Linear,
    dropout: Dropout,
}

impl Attention {
    pub fn new(config: &Config) -> Self {
        let query_matrix = Linear::new(config.embedding_dim, config.embedding_dim);
        let key_matrix = Linear::new(config.embedding_dim, config.embedding_dim);
        let value_matrix = Linear::new(config.embedding_dim, config.embedding_dim);
        let output_matrix = Linear::new(config.embedding_dim, config.embedding_dim);
        let dropout = Dropout::new(config.dropout_rate);

        Self {
            query_matrix,
            key_matrix,
            value_matrix,
            output_matrix,
            dropout,
        }
    }

    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let batch_size = input.len();
        let seq_len = input[0].len();
        let embedding_dim = self.query_matrix.output_size;
        let num_heads = embedding_dim / 64;

        let queries = self.query_matrix.forward(input);
        let keys = self.key_matrix.forward(input);
        let values = self.value_matrix.forward(input);

        let mut scores = vec![vec![vec![0.0; seq_len]; seq_len]; batch_size];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let query = &queries[b][i * embedding_dim + h * 64..(i + 1) * embedding_dim + h * 64];
                        let key = &keys[b][j * embedding_dim + h * 64..(j + 1) * embedding_dim + h * 64];
                        scores[b][i][j] += Self::dot_product(query, key) / (embedding_dim as f64).sqrt();
                    }
                }
            }
        }

        let weights = scores.iter().map(|s| Self::softmax(s)).collect::<Vec<_>>();
        let dropped_weights = self.dropout.forward(&weights);

        let mut weighted_values = vec![vec![vec![0.0; embedding_dim]; seq_len]; batch_size];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let value = &values[b][j * embedding_dim + h * 64..(j + 1) * embedding_dim + h * 64];
                        let weight = dropped_weights[b][i][j];
                        for k in 0..64 {
                            weighted_values[b][i][h * 64 + k] += value[k] * weight;
                        }
                    }
                }
            }
        }

        let concat_values = weighted_values.iter().map(|v| v.concat()).collect::<Vec<_>>();
        self.output_matrix.forward(&concat_values)
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let batch_size = grad_output.len();
        let seq_len = grad_output[0].len() / self.output_matrix.output_size;
        let embedding_dim = self.query_matrix.output_size;
        let num_heads = embedding_dim / 64;

        let grad_concat_values = self.output_matrix.backward(grad_output);
        let mut grad_weighted_values = vec![vec![vec![0.0; 64]; seq_len]; batch_size * num_heads];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let start = i * embedding_dim + h * 64;
                    let end = start + 64;
                    grad_weighted_values[b * num_heads + h][i] = grad_concat_values[b][start..end].to_vec();
                }
            }
        }

        let mut grad_values = vec![vec![0.0; embedding_dim]; batch_size * seq_len];
        let mut grad_dropped_weights = vec![vec![vec![0.0; seq_len]; seq_len]; batch_size];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let grad_weighted_value = &grad_weighted_values[b * num_heads + h][i];
                        let value = &self.value_matrix.output[b][j * embedding_dim + h * 64..(j + 1) * embedding_dim + h * 64];
                        for k in 0..64 {
                            grad_values[b * seq_len + j][h * 64 + k] += grad_weighted_value[k] * self.dropout.mask[b][i][j];
                            grad_dropped_weights[b][i][j] += grad_weighted_value[k] * value[k];
                        }
                    }
                }
            }
        }

        let grad_weights = self.dropout.backward(&grad_dropped_weights);
        let grad_scores = grad_weights.iter().map(|s| Self::softmax_backward(s)).collect::<Vec<_>>();

        let mut grad_queries = vec![vec![0.0; embedding_dim]; batch_size * seq_len];
        let mut grad_keys = vec![vec![0.0; embedding_dim]; batch_size * seq_len];
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let grad_score = grad_scores[b][i][j] / (embedding_dim as f64).sqrt();
                        let query = &self.query_matrix.output[b][i * embedding_dim + h * 64..(i + 1) * embedding_dim + h * 64];
                        let key = &self.key_matrix.output[b][j * embedding_dim + h * 64..(j + 1) * embedding_dim + h * 64];
                        for k in 0..64 {
                            grad_queries[b * seq_len + i][h * 64 + k] += grad_score * key[k];
                            grad_keys[b * seq_len + j][h * 64 + k] += grad_score * query[k];
                        }
                    }
                }
            }
        }

        let grad_input = self.query_matrix.backward(&grad_queries);
        self.key_matrix.backward(&grad_keys);
        self.value_matrix.backward(&grad_values);

        grad_input
    }

    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2).map(|(&x, &y)| x * y).sum()
    }

    fn softmax(scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let max_scores = scores.iter().map(|s| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect::<Vec<_>>();
        let exp_scores = scores.iter().zip(&max_scores).map(|(s, &m)| s.iter().map(|&x| (x - m).exp()).collect::<Vec<_>>()).collect::<Vec<_>>();
        let sum_exp_scores = exp_scores.iter().map(|s| s.iter().sum::<f64>()).collect::<Vec<_>>();
        exp_scores.iter().zip(&sum_exp_scores).map(|(s, &d)| s.iter().map(|&x| x / d).collect()).collect()
    }

    fn softmax_backward(grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let softmax_output = Self::softmax(grad_output);
        grad_output.iter().zip(&softmax_output).map(|(go, so)| so.iter().zip(go).map(|(&s, &g)| s * (g - Self::dot_product(so, go))).collect()).collect()
    }
}