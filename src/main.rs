use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, num_traits::Pow};
use std::sync::Arc;
use std::{f32::consts::PI, time::Instant};
use rustfft::FftPlanner;
use image::{imageops, GenericImageView, GrayImage, Luma};
use rustfft::Fft;
use rayon::prelude::*;

const EPS: f32 = 1e-5; // Normalization 
const RATE: f32 = 0.2; // Learning rate parameter
const THRESHOLD: f32 = 5.7; // Detection threshold

type ComplexF32 = Complex<f32>;
type FftF32 = dyn Fft<f32>;

pub struct TrackerMOSSE {
    window_size: (usize, usize),
    hann_window: Vec<f32>,
    g: Vec<ComplexF32>,
    h: Vec<ComplexF32>,
    a: Vec<ComplexF32>,
    b: Vec<ComplexF32>,
    fwd_fft: Arc<FftF32>,
    inv_fft: Arc<FftF32>,
}

impl TrackerMOSSE {
    pub fn new(planner: &mut FftPlanner<f32>, w: usize, h: usize) -> Self {
        let window_size = (w, h);
        let length = w * h;

        // Hann window to dampen signal edges
        // Combats Heisenberg uncertainty

        let value_cos = |coef: usize, v: usize| (2.0 * PI * v as f32 / coef as f32).cos(/**/);
        let cols: Vec<f32> = (0..w).map(|v| 1.0 - value_cos(w - 1, v) / 2f32).collect(/**/);

        let mut hann_window = vec![0f32; length];
        hann_window.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let row = 1.0 - value_cos(h - 1, yrow.0) / 2f32;
            (0..w).for_each(|v| yrow.1[v] = cols[v] * row);
        });

        // Complex Gaussian
        // Peaks at bounding box center
        // Represents an ideal response

        let mut gaussian = vec![0f32; length];
        let window_center = (w as f32 / 2f32, h as f32 / 2f32);
        gaussian.par_chunks_mut(w).enumerate(/**/).for_each(|yrow| {
            let dy = (yrow.0 as f32 - window_center.1).powi(2);

            for x in 0..w {
                let dx = (x as f32 - window_center.0).powi(2);
                yrow.1[x] = (-(dx + dy) / 2f32).exp(/**/);
            }
        });

        let max_value = gaussian.iter(/**/).cloned(/**/).fold(0f32, f32::max);
        let real2complex = |real_value_part| Complex::new(real_value_part / max_value, 0f32);
        let mut complex: Vec<ComplexF32> = gaussian.par_iter(/**/).map(real2complex).collect(/**/);

        let fwd_fft = planner.plan_fft_forward(length);
        let inv_fft = planner.plan_fft_inverse(length);
        fwd_fft.process(&mut complex);

        Self {
            window_size,
            hann_window,
            g: complex,
            h: vec![Complex::zero(/**/); length],
            a: vec![Complex::zero(/**/); length],
            b: vec![Complex::zero(/**/); length],
            fwd_fft,
            inv_fft,
        }
    }

    fn normalize_inplace(&self, patch: &mut [f32], bias: f32) {
        patch.par_iter_mut(/**/).for_each(|val| *val = (*val + 1f32).ln(/**/));
        let mean: f32 = patch.par_iter(/**/).sum::<f32>(/**/) / patch.len(/**/) as f32;
        let var: f32 = patch.par_iter(/**/).map(|val| (*val - mean).powi(2)).sum(/**/);
        let std: f32 = (var / patch.len(/**/) as f32).sqrt(/**/) + bias;

        patch.par_iter_mut(/**/).enumerate(/**/).for_each(|nv| {
            *nv.1 = (*nv.1 - mean) / std * self.hann_window[nv.0];
        });
    }
}

// Extract Grayscale patch pixel intensities as a 1-dimensional vector of f32
fn extract(image: &GrayImage, center: (f32, f32), width: u32, height: u32) -> Vec<f32> {
    let x = (center.0 - width as f32 / 2f32).max(0f32) as u32;
    let y = (center.1 - height as f32 / 2f32).max(0f32) as u32;
    let sub = imageops::crop_imm(image, x, y, width, height);
    sub.pixels(/**/).map(|px| px.2[0] as f32).collect(/**/)
}

fn main() {
    let mut planner = FftPlanner::new(/**/);
    let time = Instant::now();
    TrackerMOSSE::new(&mut planner, 640, 640);
    TrackerMOSSE::new(&mut planner, 640, 640);
    println!("{:?}", time.elapsed());
}
