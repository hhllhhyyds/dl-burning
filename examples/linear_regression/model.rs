use burn::{
    module::Module,
    nn::{
        self,
        loss::{MseLoss, Reduction},
    },
    tensor::backend::{AutodiffBackend, Backend},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::{feature_count, SyntheticRegressionBatch};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub net: nn::Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let net = nn::LinearConfig::new(feature_count(), 1)
            .with_bias(true)
            .init(device);
        Self { net }
    }

    pub fn forward(&self, item: SyntheticRegressionBatch<B>) -> RegressionOutput<B> {
        let batch_size = item.y.dims()[0];
        let targets = item.y.reshape([batch_size, 1]);
        let output = self.net.forward(item.x);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<SyntheticRegressionBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: SyntheticRegressionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<SyntheticRegressionBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: SyntheticRegressionBatch<B>) -> RegressionOutput<B> {
        self.forward(item)
    }
}
