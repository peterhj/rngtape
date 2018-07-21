#![feature(core_intrinsics)]
#![feature(specialization)]

extern crate byteorder;
extern crate rand;

use distributions::*;

use byteorder::*;
use rand::prelude::*;
use rand::{Error};
use rand::distributions::*;

use std::fs::{File};
use std::intrinsics::{type_name};
use std::io::{BufReader, ErrorKind};
use std::path::{PathBuf};

pub mod distributions;

pub trait TapeDistribution<T>: Distribution<T> {
  fn read_tape(&self, tape: &mut TapeState) -> T;
}

impl<T, Dist> TapeDistribution<T> for Dist where Dist: Distribution<T> {
  default fn read_tape(&self, _tape: &mut TapeState) -> T {
    unimplemented!("type: {:?} dist: {:?}", unsafe { type_name::<T>() }, unsafe { type_name::<Dist>() });
  }
}

impl TapeDistribution<isize> for Uniform<isize> {
  fn read_tape(&self, tape: &mut TapeState) -> isize {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::RandomIntegers{ref data, low, high, ..} => {
        // TODO: test for distribution equality.
        data[tape.data_off]
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<usize> for Uniform<usize> {
  fn read_tape(&self, tape: &mut TapeState) -> usize {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::RandomIntegers{ref data, low, high, ..} => {
        // TODO: test for distribution equality.
        let x = data[tape.data_off];
        assert!(x >= 0);
        x as usize
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<isize> for AbstractUniform<isize> {
  fn read_tape(&self, tape: &mut TapeState) -> isize {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::RandomIntegers{ref data, low, high, ..} => {
        let test_dist = AbstractUniform::new_inclusive(low, high);
        assert_eq!(*self, test_dist);
        data[tape.data_off]
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<usize> for AbstractUniform<usize> {
  fn read_tape(&self, tape: &mut TapeState) -> usize {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::RandomIntegers{ref data, low, high, ..} => {
        assert!(low >= 0);
        assert!(high >= 0);
        let test_dist = AbstractUniform::new_inclusive(low as usize, high as usize);
        assert_eq!(*self, test_dist);
        let x = data[tape.data_off];
        assert!(x >= 0);
        x as usize
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<f64> for StandardNormal {
  fn read_tape(&self, tape: &mut TapeState) -> f64 {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::StandardNormal{ref data, ..} => {
        data[tape.data_off]
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<f64> for Normal {
  fn read_tape(&self, tape: &mut TapeState) -> f64 {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::Normal{ref data, mean, stddev} => {
        // TODO: test for distribution equality.
        data[tape.data_off]
      }
      _ => panic!(),
    };
    next_val
  }
}

impl TapeDistribution<f64> for AbstractNormal {
  fn read_tape(&self, tape: &mut TapeState) -> f64 {
    let next_val = match tape.ops[tape.op_ctr] {
      TapeOp::Normal{ref data, mean, stddev} => {
        let test_dist = AbstractNormal::new(mean, stddev);
        assert_eq!(*self, test_dist);
        data[tape.data_off]
      }
      _ => panic!(),
    };
    next_val
  }
}

enum TapeOp {
  RandomIntegers{
    low:    isize,
    high:   isize,
    shape:  Vec<isize>,
    data:   Vec<isize>,
  },
  StandardNormal{
    shape:  Vec<isize>,
    data:   Vec<f64>,
  },
  Normal{
    mean:   f64,
    stddev: f64,
    data:   Vec<f64>,
  },
}

impl TapeOp {
  fn data_len(&self) -> usize {
    match self {
      &TapeOp::RandomIntegers{ref data, ..} => data.len(),
      &TapeOp::StandardNormal{ref data, ..} => data.len(),
      &TapeOp::Normal{ref data, ..} => data.len(),
    }
  }
}

pub struct TapeState {
  closed:   bool,
  ops:      Vec<TapeOp>,
  op_ctr:   usize,
  data_off: usize,
}

impl TapeState {
  fn is_open(&self) -> bool {
    !self.closed
  }

  fn advance(&mut self) {
    self.data_off += 1;
    if self.data_off == self.ops[self.op_ctr].data_len() {
      self.op_ctr += 1;
      self.data_off = 0;
      if self.op_ctr == self.ops.len() {
        self.closed = true;
      }
    }
  }
}

pub struct ReplayTapeRng {
  state:    TapeState,
}

impl ReplayTapeRng {
  pub fn open(path: PathBuf) -> ReplayTapeRng {
    let file = File::open(&path).unwrap();
    let mut reader = BufReader::new(file);
    let mut magic = Vec::with_capacity(7);
    for _ in 0 .. 7 {
      magic.push(reader.read_u8().unwrap());
    }
    assert_eq!(b"RNGTAPE", &magic as &[u8]);
    let _version = reader.read_u8().unwrap();
    let mut ops = vec![];
    loop {
      let ty0 = match reader.read_u8() {
        Err(e) => match e.kind() {
          ErrorKind::UnexpectedEof => {
            break;
          }
          _ => panic!(),
        },
        Ok(x) => x,
      };
      let ty1 = reader.read_u8().unwrap();
      let ty2 = reader.read_u8().unwrap();
      let ty3 = reader.read_u8().unwrap();
      let op = match &[ty0, ty1, ty2, ty3] {
        b"Dri." => {
          let low = reader.read_i64::<LittleEndian>().unwrap() as isize;
          let high = reader.read_i64::<LittleEndian>().unwrap() as isize;
          let shape_ndims = reader.read_i64::<LittleEndian>().unwrap() as isize;
          let mut flat_len = 1;
          let mut shape = Vec::with_capacity(shape_ndims as _);
          for _ in 0 .. shape_ndims {
            let d = reader.read_i64::<LittleEndian>().unwrap() as isize;
            flat_len *= d;
            shape.push(d);
          }
          let mut data = Vec::with_capacity(flat_len as _);
          for _ in 0 .. flat_len {
            data.push(reader.read_i64::<LittleEndian>().unwrap() as isize);
          }
          TapeOp::RandomIntegers{low, high, shape, data}
        }
        b"Dsn." => {
          let shape_ndims = reader.read_i64::<LittleEndian>().unwrap() as isize;
          let mut flat_len = 1;
          let mut shape = Vec::with_capacity(shape_ndims as _);
          for _ in 0 .. shape_ndims {
            let d = reader.read_i64::<LittleEndian>().unwrap() as isize;
            flat_len *= d;
            shape.push(d);
          }
          let mut data = Vec::with_capacity(flat_len as _);
          for _ in 0 .. flat_len {
            data.push(reader.read_f64::<LittleEndian>().unwrap());
          }
          TapeOp::StandardNormal{shape, data}
        }
        _ => unimplemented!(),
      };
      ops.push(op);
    }
    let state = TapeState{
      closed:   false,
      ops:      ops,
      op_ctr:   0,
      data_off: 0,
    };
    ReplayTapeRng{state}
  }
}

impl RngCore for ReplayTapeRng {
  fn next_u32(&mut self) -> u32 {
    unimplemented!();
  }

  fn next_u64(&mut self) -> u64 {
    unimplemented!();
  }

  fn fill_bytes(&mut self, _dst: &mut [u8]) {
    unimplemented!();
  }

  fn try_fill_bytes(&mut self, _dst: &mut [u8]) -> Result<(), Error> {
    unimplemented!();
  }
}

impl Rng for ReplayTapeRng {
  fn sample<T, D: Distribution<T>>(&mut self, dist: D) -> T {
    assert!(self.state.is_open());
    let x = <D as TapeDistribution<T>>::read_tape(&dist, &mut self.state);
    self.state.advance();
    x
  }
}
