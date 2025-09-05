// RUN: %clang_cc1 -fsycl-is-device -triple nvptx -internal-isystem %S/Inputs -std=c++2b -verify %s
//
// This tests that a warning is returned when a sub group size other than 32 is
// requested in the CUDA backend via the reqd_sub_group_size() kernel attribute.
#include "sycl.hpp"

int main() {

  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class invalid_kernel>([=] [[sycl::reqd_sub_group_size(8)]] {}); // expected-warning {{attribute argument 8 is invalid and will be ignored; nvptx requires sub_group size 32}}
  });

  Q.submit([&](sycl::handler &h) {
    h.single_task<class valid_kernel>([=] [[sycl::reqd_sub_group_size(32)]] {});
  });

  return 0;
}
