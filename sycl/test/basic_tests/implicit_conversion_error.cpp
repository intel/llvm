// RUN: %clangxx -fsyntax-only %fsycl-host-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue q;
  cl::sycl::context cxt = q.get_context();
  cl::sycl::device dev = q.get_device();

  cl::sycl::context cxt2{dev};
  cl::sycl::context cxt3 = dev; // expected-error {{no viable conversion from 'cl::sycl::device' to 'cl::sycl::context'}}

  cl::sycl::queue q2{dev};
  cl::sycl::queue q3 = dev; // expected-error {{no viable conversion from 'cl::sycl::device' to 'cl::sycl::queue'}}
}
