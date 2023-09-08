// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check expected runtime exceptions for annotated USM allocations

#include "sycl/sycl.hpp"
#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestUsmKind(sycl::queue &q) {
  properties P1{usm_kind<alloc::host>};
  // when the usm_kind in input property list conflicts with the usm_kind
  // argument, an exception with error code errc::invalid is thrown
  try {
    auto APtr1 = malloc_annotated<int>(N, q, alloc::device, P1);
    assert(false && "Expected exception not raised");
  } catch (sycl::exception &e) {
    if (e.code() == sycl::errc::invalid) {
      std::cout << "Exception check passed: " << e.what() << std::endl;
    }
  }
}

void TestDeviceAspect(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto Dev = q.get_device();

  if (!Dev.has(sycl::aspect::usm_shared_allocations)) {
    try {
      auto APtr1 = malloc_shared_annotated(N, Dev, Ctx);
      assert(false && "Expected exception not raised");
    } catch (sycl::exception &e) {
      if (e.code() == sycl::errc::feature_not_supported) {
        std::cout << "Exception check passed: " << e.what() << std::endl;
      }
    }
  }
}

int main() {
  sycl::queue q;
  TestUsmKind(q);
  TestDeviceAspect(q);
  return 0;
}
