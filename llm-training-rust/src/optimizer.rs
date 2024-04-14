use crate::model::Model;

pub struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, model: &mut Model) {
        if self.m.is_empty() {
            self.m = model.parameters().iter().map(|p| vec![0.0; p.len()]).collect();
            self.v = model.parameters().iter().map(|p| vec![0.0; p.len()]).collect();
        }

        self.t += 1;

        for (p, m, v) in izip!(model.parameters_mut(), &mut self.m, &mut self.v) {
            for (p_i, m_i, v_i) in izip!(p, m, v) {
                *m_i = self.beta1 * *m_i + (1.0 - self.beta1) * *p_i;
                *v_i = self.beta2 * *v_i + (1.0 - self.beta2) * p_i.powi(2);

                let m_hat = *m_i / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = *v_i / (1.0 - self.beta2.powi(self.t as i32));

                *p_i -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}