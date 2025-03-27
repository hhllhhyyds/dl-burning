mod data;
mod model;
mod training;

use burn::backend::{
    candle::{Candle, CandleDevice},
    Autodiff,
};

fn main() {
    let device = CandleDevice::cuda(0);
    training::run::<Autodiff<Candle>>(device);
}
