use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, num_traits::Pow};
use std::sync::Arc;
use std::{f32::consts::PI, time::Instant};
use rustfft::FftPlanner;
use rustfft::Fft;
use rayon::prelude::*;

const EPS: f32 = 1e-5; // Normalization 
const RATE: f32 = 0.2; // Learning rate parameter
const THRESHOLD: f32 = 5.7; // Detection threshold

type ComplexF32 = Complex<f32>;
type FftF32 = dyn Fft<f32>;

pub struct MOSSETracker {
    window_center: (f32, f32),
    window_size: (usize, usize),
    hann_window: Vec<f32>,
    g: Vec<ComplexF32>,
    h: Vec<ComplexF32>,
    a: Vec<ComplexF32>,
    b: Vec<ComplexF32>,
    inv_fft: Arc<FftF32>,
    fft: Arc<FftF32>,
}

impl MOSSETracker {
    pub fn new(w: usize, h: usize) -> Self {
        let window_center = (w as f32 / 2f32, h as f32 / 2f32);
        let length = w * h;

        // Hann window

        let one_minus_cos = |coef: usize, v: usize| 1.0 - (2.0 * PI * v as f32 / coef as f32).cos(/**/);
        let rows: Vec<f32> = (0..h).map(|v| one_minus_cos(h - 1, v) * 0.5).collect(/**/);
        let cols: Vec<f32> = (0..w).map(|v| one_minus_cos(w - 1, v) * 0.5).collect(/**/);

        let mut hann_window = vec![0f32; length];
        hann_window.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            (0..w).for_each(|v| yrow.1[v] = rows[yrow.0] * cols[v] * 1f32);
        });

        // Complex Gaussian

        let mut gaussian = vec![0f32; length];
        gaussian.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let dy = (yrow.0 as f32 - window_center.1).powi(2);

            for x in 0..w {
                let dx = (x as f32 - window_center.0).powi(2);
                yrow.1[x] = (-(dx + dy) / 2f32).exp(/**/);
            }
        });

        let max_value = gaussian.iter(/**/).cloned(/**/).fold(0f32, f32::max);
        let complex2real = |current_value| Complex::new(current_value / max_value, 0f32);
        let mut complex: Vec<ComplexF32> = gaussian.par_iter(/**/).map(complex2real).collect(/**/);

        let mut planner = FftPlanner::new(/**/);
        let fft = planner.plan_fft_forward(length);
        let inv_fft = planner.plan_fft_inverse(length);
        fft.process(&mut complex);

        Self {
            window_center,
            window_size: (w, h),
            hann_window,
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
    MOSSETracker::new(5, 5);
}
