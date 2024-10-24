// RUN: %clangxx -fsycl -sycl-std=2020 -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -fsyntax-only -Wall -Wextra

#include <sycl/sycl.hpp>

using namespace sycl;
int main() {
  // expected-warning@+1{{'atomic64' is deprecated: use sycl::aspect::atomic64 instead}}
  sycl::info::device::atomic64 atomic_64;
  (void)atomic_64;
  return 0;
}
