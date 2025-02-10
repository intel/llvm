// UNSUPPORTED: target-amd || target-nvidia
// UNSUPPORTED-INTENDED: This test is intended for backends with SPIR-V support.
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <complex>

int main() {
  std::complex<float> t(0.5f, -0.f);
  std::complex<float> r = std::exp(t);
  std::cout << r << std::endl;
  return 1;
}
