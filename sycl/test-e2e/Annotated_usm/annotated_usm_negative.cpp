// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// Check expected runtime exception thrown for invalid input of annotated USM
// allocations. Note this test does not work on gpu because the shared
// allocation tests are expected to raise an error when the target device does
// not have the corresponding aspect, while the gpu runtime has different
// behavior

#include <sycl/sycl.hpp>

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

// Check an exception is thrown if property list contains usm_kind that
// conflicts with the usm_kind argument
template <typename T> void testUsmKindConflict(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  TEST_ERRC_INVALID(malloc_annotated<T>, N, q, alloc::device,
                    properties{usm_kind_host});
  TEST_ERRC_INVALID(malloc_annotated<T>, N, dev, Ctx, alloc::device,
                    properties{usm_kind_host});
  TEST_ERRC_INVALID(malloc_annotated, N, q, alloc::host,
                    properties{usm_kind_shared});
  TEST_ERRC_INVALID(malloc_annotated, N, dev, Ctx, alloc::host,
                    properties{usm_kind_shared});
  TEST_ERRC_INVALID(aligned_alloc_annotated<T>, 0, N, q, alloc::shared,
                    properties{usm_kind_device});
  TEST_ERRC_INVALID(aligned_alloc_annotated<T>, 0, N, dev, Ctx, alloc::shared,
                    properties{usm_kind_device});
  TEST_ERRC_INVALID(aligned_alloc_annotated, N, 0, q, alloc::host,
                    properties{usm_kind_device});
  TEST_ERRC_INVALID(aligned_alloc_annotated, N, 0, dev, Ctx, alloc::host,
                    properties{usm_kind_device});
}

// Check an exception is thrown when calling malloc shared functions on SYCL
// device that does not have shared aspect
template <typename T> void testMissingDeviceAspect(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  if (!dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated, N, q);
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated, N, dev, Ctx);
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated<T>, N, q);
    TEST_FEATURE_NOT_SUPPORTED(malloc_shared_annotated<T>, N, dev, Ctx);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, q, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, dev, Ctx, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, q, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, dev, Ctx, alloc::shared);
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, q,
                               properties{usm_kind_shared});
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated, N, dev, Ctx,
                               properties{usm_kind_shared});
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, q,
                               properties{usm_kind_shared});
    TEST_FEATURE_NOT_SUPPORTED(malloc_annotated<T>, N, dev, Ctx,
                               properties{usm_kind_shared});
  }
}

int main() {
  sycl::queue q;
  testUsmKindConflict<std::complex<double>>(q);
  testMissingDeviceAspect<std::complex<double>>(q);
  return 0;
}
