use ::*;

use rand::prelude::*;
use rand::distributions::*;
use rand::distributions::uniform::{SampleUniform};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AbstractUniform<T> {
  pub low:  T,
  pub high: T,
  pub incl: bool,
}

impl<T> AbstractUniform<T> {
  pub fn new(low: T, high: T) -> Self {
    AbstractUniform{
      low, high, incl: false,
    }
  }

  pub fn new_inclusive(low: T, high: T) -> Self {
    AbstractUniform{
      low, high, incl: true,
    }
  }
}

impl<T: SampleUniform + Clone> Distribution<T> for AbstractUniform<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    match self.incl {
      false => Uniform::new(self.low.clone(), self.high.clone()),
      true  => Uniform::new_inclusive(self.low.clone(), self.high.clone()),
    }.sample(rng)
  }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AbstractNormal {
  pub mean:     f64,
  pub std_dev:  f64,
}

impl AbstractNormal {
  pub fn new(mean: f64, std_dev: f64) -> Self {
    AbstractNormal{
      mean, std_dev,
    }
  }
}

impl Distribution<f64> for AbstractNormal {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
    Normal::new(self.mean, self.std_dev).sample(rng)
  }
}
