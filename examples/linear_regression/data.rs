use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use rand::Rng;

pub const FEATURES_COUNT: usize = 5;
pub const WEIGHTS: [f32; FEATURES_COUNT] = [2.0, -3.4, 4.3, -5.5, 6.8];
pub const BIAS: f32 = -1.2;

pub const NOISE: f32 = 0.00;
pub const NUM_TRAIN: usize = 20000;
pub const NUM_VALIDATION: usize = 20000;

#[derive(Clone, Debug)]
pub struct SyntheticRegressionItem {
    pub x: [f32; FEATURES_COUNT],
    pub y: f32,
}

#[derive(Clone, Debug)]
pub struct SyntheticRegressionDataSet {
    pub data: Vec<SyntheticRegressionItem>,
}

impl SyntheticRegressionDataSet {
    pub fn gen(count: usize) -> Self {
        let mut rng = rand::rng();
        let mut data = vec![];
        (0..count).for_each(|_| {
            let mut x = [0.0; FEATURES_COUNT];
            (0..FEATURES_COUNT).for_each(|i| x[i] = rng.random_range(0f32..1.0));
            let y = x
                .iter()
                .zip(WEIGHTS.iter())
                .map(|(x, w)| w * x)
                .sum::<f32>()
                + BIAS
                + rng.random_range(0f32..1.0) * NOISE;
            data.push(SyntheticRegressionItem { x, y });
        });
        Self { data }
    }

    pub fn train() -> Self {
        Self::gen(NUM_TRAIN)
    }

    pub fn test() -> Self {
        Self::gen(NUM_VALIDATION)
    }
}

impl Dataset<SyntheticRegressionItem> for SyntheticRegressionDataSet {
    fn get(&self, index: usize) -> Option<SyntheticRegressionItem> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone, Debug)]
pub struct SyntheticRegressionBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct SyntheticRegressionBatch<B: Backend> {
    pub x: Tensor<B, 2>,
    pub y: Tensor<B, 1>,
}

impl<B: Backend> SyntheticRegressionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<SyntheticRegressionItem, SyntheticRegressionBatch<B>>
    for SyntheticRegressionBatcher<B>
{
    fn batch(&self, items: Vec<SyntheticRegressionItem>) -> SyntheticRegressionBatch<B> {
        let x_arr = items
            .iter()
            .map(|item| TensorData::from(item.x))
            .map(|data| Tensor::<B, 1>::from_data(data.convert::<B::FloatElem>(), &self.device))
            .map(|tensor| tensor.reshape([1, FEATURES_COUNT]))
            .collect();

        let y_arr = items
            .iter()
            .map(|item| TensorData::from([item.y]))
            .map(|data| Tensor::<B, 1>::from_data(data.convert::<B::FloatElem>(), &self.device))
            .collect();

        let x = Tensor::cat(x_arr, 0);
        let y = Tensor::cat(y_arr, 0);

        SyntheticRegressionBatch { x, y }
    }
}
