// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// expected-no-diagnostics

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void func() {
  const sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                       sycl::access::target::image>
      x;
}
