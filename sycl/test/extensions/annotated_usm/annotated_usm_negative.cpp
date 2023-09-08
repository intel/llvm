// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Expected failed test for conflicting usm kind in annotated USM allocation

#include "sycl/sycl.hpp"

using namespace sycl::ext::oneapi::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestUsmKind(sycl::queue &q) {
  properties P1{usm_kind<alloc::host>};

  // clang-format off
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp:* {{static assertion failed due to requirement {{.+}}: Input property list contains conflicting USM kind.}}
  // expected-error@+1 {{no matching function for call to 'malloc_shared_annotated'}}
  auto APtr1 = malloc_shared_annotated<int, decltype(P1)>(N, q);

  properties P2{};
  // expected-error-re@sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp:* {{static assertion failed due to requirement {{.+}}: USM kind is not specified. Please specify it in the arguments or in the input property list.}}
  auto APtr2 = malloc_annotated(N, q, P2);

  // clang-format on

  free(APtr1, q);
  free(APtr2, q);
}

int main() {
  sycl::queue q;

  TestUsmKind(q);
  return 0;
}
