# rmosse

Fast MOSSE tracker intended to be used in computationally constrained environments. The idea to to use `rustfft` SIMD capabilities for Fourier transforms and `rayon` to parallelize initialization phase.