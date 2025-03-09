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
    han_window: Vec<f32>,
    g: Vec<ComplexF32>,
    h: Vec<ComplexF32>,
    a: Vec<ComplexF32>,
    b: Vec<ComplexF32>,
    inv_fft: Arc<dyn Fft<f32>>,
    fft: Arc<dyn Fft<f32>>,
}

#[inline]
fn real2complex(real: f32) -> ComplexF32 {
    Complex::new(real, 0f32)
}

impl MOSSETracker {
    pub fn new(w: usize, h: usize) -> Self {
        let han_win = Self::hanning_window(w, h);
        let gaussian = Self::gaussian_center_peak(w, h);

        let mut planner = FftPlanner::new(/**/);
        let fft = planner.plan_fft_forward(w * h);
        let inv_fft = planner.plan_fft_inverse(w * h);
        fft.process(&mut gaussian);

    }

    fn hanning_window(w: usize, h: usize) -> Vec<f32> {
        let mut window = vec![0f32; w * h];

        window.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let wy = (PI * yrow.0 as f32 / (h - 1) as f32).sin(/**/);

            for x in 0..w {
                let wx = (PI * x as f32 / (w - 1) as f32).sin(/**/);
                yrow.1[x] = wx * wy;
            }
        });

        window
    }

    fn gaussian_center_peak(w: usize, h: usize) -> Vec<ComplexF32> {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mut gaussian = vec![0f32; w * h];
        
        // Calculate gaussian distribution with a peak at center
        gaussian.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let dy = (yrow.0 as f32 - cy).powi(2);

            for x in 0..w {
                let dx = (x as f32 - cx).powi(2);
                yrow.1[x] = (-(dx + dy) / 2f32).exp(/**/);
            }
        });

        // Normalize and turn into Complex for subsequent FFT
        let max_value = gaussian.iter(/**/).cloned(/**/).fold(0./0., f32::max);
        let res = gaussian.into_par_iter(/**/).map(|v| Complex::new(v / max_value, 0f32));
        res.collect(/**/)
    }
}

fn main() {
    
}
