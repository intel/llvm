// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check expected runtime exceptions for annotated USM allocations

#include "sycl/sycl.hpp"
#include <complex>
#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

using alloc = sycl::usm::alloc;

constexpr int N = 10;

#define TEST_ERRC_INVALID(f, args...)                                          \
  try {                                                                        \
    auto ap = f(args);                                                         \
    assert(false && "Expected errc::invalid not raised");                      \
  } catch (sycl::exception & e) {                                              \
    if (e.code() == sycl::errc::invalid) {                                     \
      std::cout << "Exception check passed: " << e.what() << std::endl;        \
    }                                                                          \
  }

#define TEST_FEATURE_NOT_SUPPORTED(f, args...)                                 \
  try {                                                                        \
    auto ap = f(args);                                                         \
    assert(false &&                                                            \
           "Expected exception sycl::errc::feature_not_supported not raised"); \
  } catch (sycl::exception & e) {                                              \
    if (e.code() == sycl::errc::feature_not_supported) {                       \
      std::cout << "Exception check passed: " << e.what() << std::endl;        \
    }                                                                          \
  }

// Check an exception is thrown if property list contains usm_kind conflicts
// with the usm_kind argument
template <typename T> void testUsmKindConflict(sycl::queue &q) {
  TEST_ERRC_INVALID(malloc_annotated<T>, N, q, alloc::device,
                    properties{usm_kind_host});
  TEST_ERRC_INVALID(malloc_annotated, N, q, alloc::host,
                    properties{usm_kind_shared});
  TEST_ERRC_INVALID(aligned_alloc_annotated<T>, 0, N, q, alloc::shared,
                    properties{usm_kind_device});
  TEST_ERRC_INVALID(aligned_alloc_annotated, N, 0, q, alloc::host,
                    properties{usm_kind_device});
}

// Check an exception is thrown when calling malloc shared functions when
// the device does not have shared aspects
template <typename T> void testMissingDeviceAspect(sycl::queue &q) {
  auto Dev = q.get_device();
  if (!Dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated, N, q);
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated<T>, N, q);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, q, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, q, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, q,
                               properties{usm_kind_shared});
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, q,
                               properties{usm_kind_shared});
  }
}

int main() {
  sycl::queue q;
  testUsmKindConflict<std::complex<double>>(q);
  testMissingDeviceAspect<std::complex<double>>(q);
  return 0;
}
