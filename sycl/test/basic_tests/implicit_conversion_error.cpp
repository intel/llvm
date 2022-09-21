// RUN: %clangxx -fsyntax-only %fsycl-host-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::context cxt = q.get_context();
  sycl::device dev = q.get_device();

  // clang-format off
  sycl::context cxt2{dev};
  sycl::context cxt3 = dev; // expected-error {{no viable conversion from 'sycl::device' to 'sycl::context'}}

  sycl::queue q2{dev};
  sycl::queue q3 = dev; // expected-error {{no viable conversion from 'sycl::device' to 'sycl::queue'}}
  // clang-format on
}
