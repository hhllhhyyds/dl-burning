use std::io::Write;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::SgdConfig,
    record::{CompactRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            CpuMemory, CpuTemperature, CpuUse, CudaMetric, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::{
    data::{SyntheticRegressionBatcher, SyntheticRegressionDataSet},
    model::Model,
};

static ARTIFACT_DIR: &str = "/tmp/dl-burning-linear-regression";

#[derive(Config)]
pub struct LinearRegressionTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: SgdConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);

    // Config
    let config_optimizer = SgdConfig::new();
    let config = LinearRegressionTrainingConfig::new(config_optimizer);

    // Data
    let batcher_train = SyntheticRegressionBatcher::<B>::new(device.clone());
    let batcher_valid = SyntheticRegressionBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SyntheticRegressionDataSet::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SyntheticRegressionDataSet::test());

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(Model::new(&device), config.optimizer.init(), 0.03);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .expect("Failed to save config");

    let mut file = std::fs::File::create(format!("{ARTIFACT_DIR}/model_save.log")).unwrap();
    write!(
        file,
        "weights:\n{}\nbias:\n{}\n",
        model_trained.net.weight.val().to_string(),
        model_trained.net.bias.as_ref().unwrap().val().to_string()
    )
    .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &PrettyJsonFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save trained model");
}
