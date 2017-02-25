//! Detect the frequency of a signal.
//!
//! The frequency is calculated using the AMDF algorithm.
//!
//! See http://www.instructables.com/id/Arduino-Pitch-Detection-Algorithm-AMDF/ for details.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(test, warn(missing_docs))]
#![cfg_attr(test, deny(warnings))]

#[cfg(test)]
extern crate core;
extern crate num_traits;

use num_traits::{Num, Float, Bounded, NumCast, FromPrimitive, ToPrimitive};
use core::cmp;
use core::ops::AddAssign;
use core::marker::PhantomData;

/// A frequency.
pub type SampleRate = usize;

/// The duration of a signal's period, in number of samples.
type Period = usize;

/// A frequency analyzer.
pub struct Analyzer<F> {
    /// The sampling rate of the input signal.
    sample_rate: SampleRate,
    /// The minimum duration of the signal's period, in number of samples.
    min_period: Period,
    /// The maximum duration of the signal's period, in number of samples.
    max_period: Period,
    /// The detected duration of the signal's period, in number of samples.
    detected_period: Period,
    // Avoids error E0392 (unused type parameter)
    _unused: PhantomData<F>,
}

impl<F> Analyzer<F>
    where F: Float + FromPrimitive
{
    /// Create a new analyzer.
    ///
    /// # Arguments:
    ///
    /// - `sample_rate`: The sampling rate of the signal, in Hz.
    /// - `min_freq`: The minimum frequency to detect, in Hz.
    /// - `max_freq`: The maximum frequency to detect, in Hz.
    ///
    fn new(sample_rate: SampleRate, min_period: usize, max_period: usize) -> Self {
        Analyzer {
            sample_rate: sample_rate,
            min_period: min_period,
            max_period: max_period,
            detected_period: 0,
            _unused: PhantomData,
        }
    }

    /// Calculate the signal's frequency from samples.
    #[inline]
    pub fn analyse<N>(&mut self, samples: &[N])
        where N: Num + Bounded + AddAssign<N> + ToPrimitive + Copy
    {
        let mut min_error_rate: F = F::max_value();
        let mut period = 0;

        for i in (self.min_period)..(self.max_period) {
            let error_rate = amdf(samples, i);
            if error_rate < min_error_rate {
                min_error_rate = error_rate;
                period = i;
            }
        }

        self.detected_period = period;
    }

    /// Get the calculated frequency, in Hz.
    ///
    /// Returns `None` if the method `Analyzer::input()` hasn't been called.
    #[inline]
    pub fn get_freq(&self) -> Option<F> {
        if self.detected_period > 0 {
            Some(F::from_usize(self.sample_rate).unwrap() /
                 F::from_usize(self.detected_period).unwrap())
        } else {
            None
        }
    }
}

/// Apply the AMDF alogrithm.
#[inline]
fn amdf<N, F>(samples: &[N], period: usize) -> F
    where N: Num + AddAssign<N> + ToPrimitive + Copy,
          F: Float
{
    let mut sum = N::zero();
    let max: usize = cmp::min(period, samples.len() / 2);
    for i in 0..max {
        sum += samples[i] - samples[i + max];
    }
    F::one() / <F as NumCast>::from(period).unwrap() * <F as NumCast>::from(sum).unwrap().abs()
}


/// Create an an Analyzer.
pub struct Builder {
    sample_rate: SampleRate,
    min_freq: usize,
    max_freq: usize,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            sample_rate: 44100,
            min_freq: 150,
            max_freq: 1500,
        }
    }

    /// Set the input sample rate.
    pub fn sample_rate(mut self, sample_rate: SampleRate) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the minimum frequency to detect.
    pub fn min_freq(mut self, min_freq: usize) -> Self {
        self.min_freq = min_freq;
        self
    }

    /// Set the maximum frequency to detect.
    pub fn max_freq(mut self, max_freq: usize) -> Self {
        self.max_freq = max_freq;
        self
    }

    /// Build the analyzer.
    pub fn finalize<F>(&self) -> Result<Analyzer<F>, &'static str>
        where F: Float + FromPrimitive
    {
        if self.sample_rate == 0 {
            Err("sample rate must be greater than zero.")
        } else if self.min_freq >= self.max_freq {
            Err("the minimum frequency must be less than the maximum frequency")
        } else {
            let min_period = self.sample_rate / self.max_freq;
            let max_period = self.sample_rate / self.min_freq;
            Ok(Analyzer::new(self.sample_rate, min_period, max_period))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn create_signal(freq: f32, sample_rate: usize, size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| (freq * PI * 2f32 * (i as f32) / (sample_rate as f32)).sin())
            .collect()
    }

    #[test]
    fn it_works() {
        let size = 1024;
        let sample_rate = 44100;
        let signal: Vec<f32> = create_signal(440f32, sample_rate, size);
        let mut a = Builder::new()
            .sample_rate(sample_rate)
            .min_freq(150)
            .max_freq(1000)
            .finalize()
            .unwrap();
        a.analyse(&signal);
        assert_eq!(a.get_freq(), Some(441f32));
    }
}
