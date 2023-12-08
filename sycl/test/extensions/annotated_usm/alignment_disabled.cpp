// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Ensure the compile-time `alignment` property is disabled on annotated_ptr
// TODO: Remove this test when `aligment` is supported in annotated_ptr

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

// clang-format off

void testAlignmentDisabled(sycl::queue &q) {
    const sycl::context &Ctx = q.get_context();
    auto Dev = q.get_device();

    // expected-error@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* {{static assertion failed due to requirement '!hasAlignment': The alignment property is not supported in annotated_ptr class in oneAPI 2024.1.}}
    // expected-error@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement '!hasAlignment': The alignment property is not supported by annotated malloc functions in oneAPI 2024.1.}}
    auto p1 = malloc_annotated<int>(N, q, sycl::usm::alloc::host, properties{buffer_location<2>, alignment<1>});

    // expected-error@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* 1+ {{static assertion failed due to requirement '!hasAlignment': The alignment property is not supported in annotated_ptr class in oneAPI 2024.1.}}
    // expected-error@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement '!hasAlignment': The alignment property is not supported by annotated malloc functions in oneAPI 2024.1.}}
    auto p2 = malloc_device_annotated(N, q, properties{alignment<2>});

    q.single_task<class MyIP>([=] { *p1 += *p2; }).wait();
  }

// clang-format on

int main() {
  sycl::queue q;
  testAlignmentDisabled(q);
  return 0;
}
