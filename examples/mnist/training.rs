use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    tensor::backend::AutodiffBackend,
};

use crate::data::MnistBatcher;

static ARTIFACT_DIR: &str = "/tmp/dl-burning-mnist";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);

    // TODO: config
    let config = MnistTrainingConfig::new();
    B::seed(config.seed);

    // Data
    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());
}

fn main() {
    println!("mnist example");
}
