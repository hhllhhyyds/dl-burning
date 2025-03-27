use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Clone, Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert::<B::FloatElem>(), &self.device))
            .map(|tensor| tensor.reshape([1, WIDTH, HEIGHT]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| TensorData::from([(item.label as i64).elem::<B::IntElem>()]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
