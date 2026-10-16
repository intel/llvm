// Check that a -fno-sycl-rdc object built for two GPU architectures carries
// the device images of both and runs on either device.

// REQUIRES: ocloc, target-spir
// REQUIRES: arch-intel_gpu_acm_g10 || arch-intel_gpu_bmg_g21

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g10,intel_gpu_bmg_g21 -fno-sycl-rdc -c %S/Inputs/simple.cpp -o %t.o
// RUN: %clangxx -fsycl %t.o -o %t.out
// RUN: %{run} %t.out
