// REQUIRES: cuda && hip
// Note: This isn't really target specific and should be switched to spir when
// it's enabled for it.

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S -emit-llvm %s -o - | FileCheck %s

#include <cmath>
#include <complex>

// CHECK-LABEL: test
__attribute__((sycl_device)) void test(std::complex<float> *cfp,
                                       std::complex<double> *cdp,
                                       std::complex<float> *rcf,
                                       std::complex<double> *rcd) {
  // Use an incrementing index to prevent the compiler from optimizing some
  // calls that would store to the same address.
  int idx = 0;

  // TODO: Check that calls to the functions with std::complex parameters only
  // produce calls to SPIR-V or LLVM intrinsics and no other calls.

  // For now just check that there no calls to non-mangled functions starting
  // with `c` (i.e. prefix for C complex functions).
  // CHECK-NOT: call {{.*}}@c

  rcf[idx++] = cfp[0] + cfp[1];
  rcd[idx++] = cdp[0] + cdp[1];

  rcf[idx++] = cfp[0] - cfp[1];
  rcd[idx++] = cdp[0] - cdp[1];

  rcf[idx++] = cfp[0] * cfp[1];
  rcd[idx++] = cdp[0] * cdp[1];

  rcf[idx++] = cfp[0] / cfp[1];
  rcd[idx++] = cdp[0] / cdp[1];

  rcf[idx++] = std::imag(cfp[0]);
  rcd[idx++] = std::imag(cdp[0]);

  rcf[idx++] = std::real(cfp[0]);
  rcd[idx++] = std::real(cdp[0]);

  rcf[idx++] = std::abs(cfp[0]);
  rcd[idx++] = std::abs(cdp[0]);

  rcf[idx++] = std::arg(cfp[0]);
  rcd[idx++] = std::arg(cdp[0]);

  rcf[idx++] = std::norm(cfp[0]);
  rcd[idx++] = std::norm(cdp[0]);

  rcf[idx++] = std::conj(cfp[0]);
  rcd[idx++] = std::conj(cdp[0]);

  rcf[idx++] = std::proj(cfp[0]);
  rcd[idx++] = std::proj(cdp[0]);

  // FIXME: libstdc++ implementation of std::polar compares std::complex<float>
  // with an int, which is not suppoted.
  // rcf[idx++] = std::polar(cfp[0]);
  // rcd[idx++] = std::polar(cdp[0]);

  rcf[idx++] = std::exp(cfp[0]);
  rcd[idx++] = std::exp(cdp[0]);

  rcf[idx++] = std::log(cfp[0]);
  rcd[idx++] = std::log(cdp[0]);

  rcf[idx++] = std::log10(cfp[0]);
  rcd[idx++] = std::log10(cdp[0]);

  rcf[idx++] = std::pow(cfp[0], cfp[1]);
  rcd[idx++] = std::pow(cdp[0], cdp[1]);

  rcf[idx++] = std::sqrt(cfp[0]);
  rcd[idx++] = std::sqrt(cdp[0]);

  rcf[idx++] = std::sin(cfp[0]);
  rcd[idx++] = std::sin(cdp[0]);

  rcf[idx++] = std::cos(cfp[0]);
  rcd[idx++] = std::cos(cdp[0]);

  rcf[idx++] = std::tan(cfp[0]);
  rcd[idx++] = std::tan(cdp[0]);

  rcf[idx++] = std::asin(cfp[0]);
  rcd[idx++] = std::asin(cdp[0]);

  rcf[idx++] = std::acos(cfp[0]);
  rcd[idx++] = std::acos(cdp[0]);

  rcf[idx++] = std::atan(cfp[0]);
  rcd[idx++] = std::atan(cdp[0]);

  rcf[idx++] = std::sinh(cfp[0]);
  rcd[idx++] = std::sinh(cdp[0]);

  rcf[idx++] = std::cosh(cfp[0]);
  rcd[idx++] = std::cosh(cdp[0]);

  rcf[idx++] = std::tanh(cfp[0]);
  rcd[idx++] = std::tanh(cdp[0]);

  rcf[idx++] = std::asinh(cfp[0]);
  rcd[idx++] = std::asinh(cdp[0]);

  rcf[idx++] = std::acosh(cfp[0]);
  rcd[idx++] = std::acosh(cdp[0]);

  rcf[idx++] = std::atanh(cfp[0]);
  rcd[idx++] = std::atanh(cdp[0]);
}
