pub trait TensorNumber: Copy {
    fn zero() -> Self;
    fn one() -> Self;
}

impl TensorNumber for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

/// Fundamental tensor type.
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T: TensorNumber> Tensor<T> {
    /// Fill a new Tensor with the value, using the given shape.
    pub fn fill(shape: &[usize], value: T) -> Self {
        let size = shape.iter().product();
        Self {
            data: (0..size).map(|_| value).collect(),
            shape: shape.to_vec(),
        }
    }

    /// Create a new Tensor with the shape, with 1.0 set along the diagonal.
    pub fn eye(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let data = (0..size).map(|_| T::zero()).collect();
        let mut tensor = Self {
            data,
            shape: shape.to_vec(),
        };
        let max = *shape.iter().max().unwrap();
        let mut ti: Vec<usize> = shape.iter().map(|_| 0).collect();
        for i in 0..max {
            ti.fill(i);
            tensor.set(&ti, T::one());
        }
        tensor
    }

    /// Set the value at the given tensor index.
    fn set(&mut self, ti: &[usize], value: T) {
        let (true_idx, _) = ti
            .iter()
            .enumerate()
            .fold((0, 1), |(sum, fac), (pos, subi)| {
                let next_fac = if pos > 0 {
                    fac * self.shape[pos - 1]
                } else {
                    fac
                };
                (sum + subi * next_fac, next_fac)
            });
        self.data[true_idx] = value;
    }
}
