use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, num_traits::Pow};
use std::sync::Arc;
use std::{f32::consts::PI, time::Instant};
use rustfft::FftPlanner;
use rustfft::Fft;
use rayon::prelude::*;

const EPS: f32 = 1e-5; // Normalization 
const RATE: f32 = 0.2; // Learning rate parameter
const PSR_THRESHOLD: f32 = 5.7; // Detection threshold

type ComplexF32 = Complex<f32>;

pub struct MOSSETracker {
    window_center: (f32, f32),
    window_size: (usize, usize),
    hanning_window: Vec<f32>,
    g: Vec<ComplexF32>,
    h: Vec<ComplexF32>,
    a: Vec<ComplexF32>,
    b: Vec<ComplexF32>,
    inv_fft: Arc<dyn Fft<f32>>,
    fft: Arc<dyn Fft<f32>>,
}

impl MOSSETracker {
    pub fn new(w: usize, h: usize) -> Self {
        let window_center = (w as f32 / 2f32, h as f32 / 2f32);
        let length = w * h;

        let mut hanning_window = vec![0f32; length];
        let mut gaussian = vec![0f32; length];
        
        // TODO: precompute for max possible w * h, then take slices
        hanning_window.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let wy = (PI * yrow.0 as f32 / (h - 1) as f32).sin(/**/);

            for x in 0..w {
                let wx = PI * x as f32 / (w - 1) as f32;
                yrow.1[x] = wx.sin(/**/) * wy;
            }
        });

        // Calculate gaussian distribution with a peak at center
        gaussian.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let dy = (yrow.0 as f32 - window_center.1).powi(2);

            for x in 0..w {
                let dx = (x as f32 - window_center.0).powi(2);
                yrow.1[x] = (-(dx + dy) / 2f32).exp(/**/);
            }
        });

        
        let max_value = gaussian.iter(/**/).cloned(/**/).fold(0./0., f32::max);
        let complex2real = |current_value| Complex::new(current_value / max_value, 0f32);
        let mut complex: Vec<ComplexF32> = gaussian.par_iter(/**/).map(complex2real).collect(/**/);

        let mut planner = FftPlanner::new(/**/);
        let fft = planner.plan_fft_forward(length);
        let inv_fft = planner.plan_fft_inverse(length);
        fft.process(&mut complex);

        Self {
            window_center,
            window_size: (w, h),
            hanning_window,
            g: complex,
            h: vec![Complex::zero(/**/); length],
            a: vec![Complex::zero(/**/); length],
            b: vec![Complex::zero(/**/); length],
            inv_fft,
            fft,
        }
    }
}

fn main() {
    MOSSETracker::new(500, 500);
}
