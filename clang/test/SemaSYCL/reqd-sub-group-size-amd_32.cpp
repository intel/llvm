// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx1010 -internal-isystem %S/Inputs -std=c++2b -verify %s

// Sub-group size is optimized for 32, warn (and ignore the attribute) if the
// size is not 32.
#include "sycl.hpp"

int main() {

  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class invalid_kernel>([=] [[sycl::reqd_sub_group_size(64)]] {}); // expected-warning {{attribute argument 64 is invalid and will be ignored; amdgcn requires sub_group size 32}}
  });

  Q.submit([&](sycl::handler &h) {
    h.single_task<class valid_kernel>([=] [[sycl::reqd_sub_group_size(32)]] {});
  });

  Q.submit([&](sycl::handler &h) {
    h.single_task<class invalid_kernel_2>([=] [[sycl::reqd_sub_group_size(8)]] {}); // expected-warning {{attribute argument 8 is invalid and will be ignored; amdgcn requires sub_group size 32}}
  });

  return 0;
}
