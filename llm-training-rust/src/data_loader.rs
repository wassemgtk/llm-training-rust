use crate::tokenizer::Tokenizer;

pub struct DataLoader {
    data: Vec<usize>,
    pub batch_size: usize,
    pub seq_len: usize,
    idx: usize,
}

impl DataLoader {
    pub fn new(file_path: &str, batch_size: usize, seq_len: usize, tokenizer: &Tokenizer) -> Self {
        let text = std::fs::read_to_string(file_path).expect("Failed to read data file");
        let data = tokenizer.encode(&text);
        Self {
            data,
            batch_size,
            seq_len,
            idx: 0,
        }
    }

    pub fn iter(&mut self) -> DataLoaderIter {
        DataLoaderIter {
            data_loader: self,
            idx: 0,
        }
    }

    pub fn len(&self) -> usize {
        (self.data.len() - 1) / (self.batch_size * self.seq_len)
    }
}

pub struct DataLoaderIter<'a> {
    data_loader: &'a mut DataLoader,
    idx: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.data_loader.data.len() - 1 {
            self.data_loader.idx = 0;
            self.idx = 0;
        }

        let batch_size = self.data_loader.batch_size;
        let seq_len = self.data_loader.seq_len;

        let mut batch_input = Vec::with_capacity(batch_size * seq_len);
        let mut batch_target = Vec::with_capacity(batch_size * seq_len);

        for _ in 0..batch_size {
            let start = self.idx;
            let end = start + seq_len;
            batch_input.extend_from_slice(&self.data_loader.data[start..end]);
            batch_target.extend_from_slice(&self.data_loader.data[start + 1..=end]);
            self.idx += seq_len;
        }

        Some((batch_input, batch_target))
    }
}