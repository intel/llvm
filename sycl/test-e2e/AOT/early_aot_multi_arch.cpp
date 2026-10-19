// Check that a -fno-sycl-rdc object built for two GPU architectures carries
// the device images of both and runs on either device.

// REQUIRES: ocloc, target-spir
// REQUIRES: arch-intel_gpu_acm_g10 || arch-intel_gpu_bmg_g21

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g10,intel_gpu_bmg_g21 -fno-sycl-rdc -c %S/Inputs/simple.cpp -o %t.o
// With the new offloading model the object already contains the wrapped device
// images, so a plain host link is enough. The old model extracts them from the
// object's offload bundle at a -fsycl link. On Windows, -fsycl also selects the
// dynamic C runtime that the object was compiled for.
// RUN: %clangxx %if new-offload-model && linux %{ -Wno-unused-command-line-argument %t.o -o %t.out %sycl_options %} %else %{ -fsycl %t.o -o %t.out %}
// RUN: %{run} %t.out
