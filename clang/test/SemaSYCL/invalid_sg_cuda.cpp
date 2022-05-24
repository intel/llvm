// RUN: %clang_cc1 -fsycl-is-device -triple nvptx -internal-isystem %S/Inputs -std=c++2b -verify %s

#include "sycl.hpp"

int main() {

  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class kernel_name>([=] [[sycl::reqd_sub_group_size(8)]] { // expected-warning {{CUDA requires sub_group size 32: size 8 request will be ignored}}

    });
  });

  return 0;
}
