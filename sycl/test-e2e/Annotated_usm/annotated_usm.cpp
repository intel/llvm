// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// E2E tests for annotated USM allocation functions

// clang-format off

#include "sycl/sycl.hpp"
#include <complex>
#include "<numeric>"

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

// Single test instance consisting of (i) calling malloc function `f` (ii)
// checking usm location of returned pointer  (iii) calling free
#define TEST_USM_KIND(f, kind, args...)                                        \
  {                                                                            \
    auto ap = f(args);                                                         \
    assert(ap.get() != nullptr &&                                              \
           sycl::get_pointer_type(ap.get(), Ctx) == kind);                     \
    free(ap, q);                                                               \
  }

// Single test instance consisting of (i) calling malloc function `f` (ii)
// checking alignment Note that the annotated_ptr is not freed in alignment
// tests to avoid the same base address being returned every time
#define TEST_ALIGN(f, align, args...)                                          \
  {                                                                            \
    auto ap = f(args);                                                         \
    assert(ap.get() != nullptr && ((uintptr_t)ap.get() % N) == 0);             \
  }

// Single test instance consisting of (i) calling malloc function `f` (ii)
// checking return is nullptr (iii) calling free
#define TEST_ALIGN_NULL_EXPECTED(f, args...)                                   \
  {                                                                            \
    auto ap = f(args);                                                         \
    assert(ap.get() == nullptr);                                               \
    free(ap, q);                                                               \
  }

constexpr int N = 10;

template <typename T> void testUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  TEST_USM_KIND(malloc_annotated<T>, alloc::device, N, q, alloc::device);
  TEST_USM_KIND(malloc_annotated, alloc::device, N, q, alloc::device);
  TEST_USM_KIND(aligned_alloc_annotated<T>, alloc::host, 1, N, q, alloc::host);
  TEST_USM_KIND(aligned_alloc_annotated, alloc::host, 1, N, q, alloc::host);

  TEST_USM_KIND(malloc_device_annotated, alloc::device, N, q);
  TEST_USM_KIND(malloc_device_annotated<T>, alloc::device, N, q);
  TEST_USM_KIND(aligned_alloc_device_annotated, alloc::device, 0, N, q);
  TEST_USM_KIND(aligned_alloc_device_annotated<T>, alloc::device, 0, N, q);

  TEST_USM_KIND(malloc_host_annotated, alloc::host, N, q);
  TEST_USM_KIND((malloc_host_annotated<T>), alloc::host, N, q);
  TEST_USM_KIND(aligned_alloc_host_annotated, alloc::host, 0, N, q);
  TEST_USM_KIND((aligned_alloc_host_annotated<T>), alloc::host, 0, N, q);

  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_USM_KIND(malloc_annotated, alloc::shared, N, q,
                  properties{usm_kind_shared});
    TEST_USM_KIND(malloc_annotated<T>, alloc::shared, N, q,
                  properties{usm_kind_shared});

    TEST_USM_KIND(malloc_shared_annotated, alloc::shared, N, q);
    TEST_USM_KIND((malloc_shared_annotated<T>), alloc::shared, N, q);
    TEST_USM_KIND(aligned_alloc_shared_annotated, alloc::shared, 0, N, q);
    TEST_USM_KIND((aligned_alloc_shared_annotated<T>), alloc::shared, 0, N, q);
  }
}

template <typename T> void testAlign(sycl::queue &q, unsigned align) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  assert(align > 0 || (align & (align - 1)) == 0);

  // Case: malloc_xxx with no alignment property, the alignment is sizeof(T)
  TEST_ALIGN((malloc_device_annotated<T>), sizeof(T), N, q);
  TEST_ALIGN((malloc_host_annotated<T>), sizeof(T), N, q);
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN((malloc_shared_annotated<T>), sizeof(T), N, q);
  }

  // Case: malloc_xxx with compile-time alignment<0>, the alignment is sizeof(T)
  TEST_ALIGN((malloc_device_annotated<T>), sizeof(T), N, q,
             properties{alignment<0>});
  TEST_ALIGN((malloc_host_annotated<T>), sizeof(T), N, q,
             properties{alignment<0>});
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN((malloc_shared_annotated<T>), sizeof(T), N, q,
               properties{alignment<0>});
  }

  // Case: malloc_xxx with compile-time alignment<N> (N is not a power of 2),
  // nullptr is returned
  TEST_ALIGN_NULL_EXPECTED((malloc_device_annotated<T>), N, q,
                           properties{alignment<3>});
  TEST_ALIGN_NULL_EXPECTED(malloc_device_annotated, N, q,
                           properties{alignment<7>});
  TEST_ALIGN_NULL_EXPECTED((malloc_host_annotated<T>), N, q,
                           properties{alignment<15>});
  TEST_ALIGN_NULL_EXPECTED(malloc_host_annotated, N, q,
                           properties{alignment<7>});
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN_NULL_EXPECTED((malloc_shared_annotated<T>), N, q,
                             properties{alignment<31>});
    TEST_ALIGN_NULL_EXPECTED(malloc_shared_annotated, N, q,
                             properties{alignment<63>});
  }

  // Case: malloc_xxx with compile-time alignment<N> (N is a power of 2), the
  // alignment is the least-common-multiple of N and sizeof(T)
  TEST_ALIGN((malloc_device_annotated<T>), std::lcm(2, sizeof(T)), N, q,
             properties{alignment<2>});
  TEST_ALIGN(malloc_device_annotated, 2, N, q, properties{alignment<2>});
  TEST_ALIGN((malloc_host_annotated<T>), std::lcm(8, sizeof(T)), N, q,
             properties{alignment<8>});
  TEST_ALIGN(malloc_host_annotated, 8, N, q, properties{alignment<8>});
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN((malloc_shared_annotated<T>), std::lcm(16, sizeof(T)), N, q,
               properties{alignment<16>});
    TEST_ALIGN(malloc_shared_annotated, 16, N, q, properties{alignment<16>});
  }

  // Case: aligned_alloc_xxx with no alignment property, and the alignment
  // argument is 0 the result is sizeof(T)-aligned
  TEST_ALIGN((aligned_alloc_device_annotated<T>), sizeof(T), 0, N, q);
  TEST_ALIGN((aligned_alloc_host_annotated<T>), sizeof(T), 0, N, q);
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN((aligned_alloc_shared_annotated<T>), sizeof(T), 0, N, q);
  }

  // Case: aligned_alloc_xxx with no alignment property, and the alignment
  // argument is not a power of 2, the result is nullptr
  TEST_ALIGN_NULL_EXPECTED(aligned_alloc_device_annotated, 3, N, q);
  TEST_ALIGN_NULL_EXPECTED((aligned_alloc_host_annotated<T>), 7, N, q);
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN_NULL_EXPECTED((aligned_alloc_shared_annotated<T>), 15, N, q);
  }

  // Case: aligned_alloc_xxx with compile-time alignment<N> (N is a power of 2),
  // the alignment is the least-common-multiple of the alignment argument (a
  // power of 2), N and sizeof(T)
  TEST_ALIGN((aligned_alloc_device_annotated<T>),
             std::lcm(align, std::lcm(2, sizeof(T))), align, N, q,
             properties{alignment<2>});
  TEST_ALIGN((aligned_alloc_host_annotated<T>),
             std::lcm(align, std::lcm(8, sizeof(T))), align, N, q,
             properties{alignment<8>});
  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    TEST_ALIGN((aligned_alloc_shared_annotated<T>),
               std::lcm(align, std::lcm(32, sizeof(T))), align, N, q,
               properties{alignment<32>});
  }
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);

  testUsmKind<int>(q);
  return 0;
}