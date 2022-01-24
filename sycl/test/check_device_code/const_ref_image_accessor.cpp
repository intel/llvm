// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// expected-no-diagnostics

#include <sycl/sycl.hpp>

void func(sycl::handler &handler, const sycl::range<2> &range,
          const sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                               sycl::access::target::image> &dummy_accessor) {
  handler.parallel_for(range, [=](const sycl::item<2> item) {
    const sycl::int2 coords{item[1], item[0]};
    auto pixel = dummy_accessor.read(coords);
  });
}
