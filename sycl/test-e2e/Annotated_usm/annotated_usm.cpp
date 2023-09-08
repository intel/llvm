// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// E2E test for annotated USM allocation

#include "sycl/sycl.hpp"
#include <iostream>

// clang-format on

#define CHECK_USM_KIND(ap, Kind)                                               \
  assert(ap.get() != nullptr && sycl::get_pointer_type(ap.get(), Ctx) == Kind)
#define CHECK_ALIGN(ap, N)                                                     \
  assert(ap.get() != nullptr && ((uintptr_t)ap.get() % N) == 0)

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  // Allocation funtions with the USM kind in the function name
  properties P1{conduit, stable};
  auto APtr1 = malloc_device_annotated<int>(N, q, P1);
  CHECK_USM_KIND(APtr1, alloc::device);

  auto APtr2 = aligned_alloc_device_annotated(128, N, q);
  CHECK_USM_KIND(APtr2, alloc::device);

  auto APtr3 = malloc_host_annotated<int>(N, q);
  CHECK_USM_KIND(APtr3, alloc::host);

  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    auto APtr4 = malloc_shared_annotated(N, q);
    CHECK_USM_KIND(APtr4, alloc::shared);
    free(APtr4, q);
  }

  // Parameterized functions: the USM kind is specified by an argument
  properties P2{conduit, cache_config{large_slm}};
  auto APtr5 = malloc_annotated<int>(N, q, alloc::device, P2);
  CHECK_USM_KIND(APtr5, alloc::device);

  // Functions where the USM kind is required in the propList
  properties P3{usm_kind<alloc::host>};
  auto APtr6 = malloc_annotated<int>(N, q, P3);
  CHECK_USM_KIND(APtr6, alloc::host);

  // Call sycl::free() on the underlying raw pointer of annotated_ptr
  sycl::free(APtr1.get(), q);
  sycl::free(APtr6.get(), q);

  // Call new annotated dealloc function directly on annotated_ptr
  free(APtr2, q);
  free(APtr3, q);
  free(APtr5, q);
}

void TestAlign(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  properties AL0{alignment<0>};
  properties AL2{alignment<2>};
  properties AL8{alignment<8>};
  properties AL64{alignment<64>};
  properties AL512{alignment<512>};

  //------ Test allocations in bytes

  // alignment<N> property: the allocated raw pointer should be aligned by N
  auto APtr1 = malloc_annotated(N, q, alloc::device, AL512);
  CHECK_ALIGN(APtr1, 512);

  // alignment arg N1 and alignment<N2> property (N1 > 0, N2 > 0)
  // the allocated raw pointer should be aligned by lcm(N1, N2)
  auto APtr12 = aligned_alloc_device_annotated(512 /* alignment */, N, q, AL64);
  CHECK_ALIGN(APtr12, 512);
  auto APtr13 = aligned_alloc_device_annotated(2 /* alignment */, N, q, AL64);
  CHECK_ALIGN(APtr13, 64);

  // alignment arg N1 and alignment<N2> property (N1 = 0, N2 > 0)
  // the allocated raw pointer should be N2-byte aligned
  auto APtr14 = aligned_alloc_host_annotated(0, N, q, AL8);
  CHECK_ALIGN(APtr14, 8);

  // alignment arg N1 and alignment<N2> property (N1 > 0, N2 = 0)
  // the allocated raw pointer should be N1-byte aligned
  auto APtr15 = aligned_alloc_host_annotated(8, N, q, AL0);
  CHECK_ALIGN(APtr15, 8);

  //------- Test allocations in elements of type T

  // alignment<N> (N>0): the raw pointer should be aligned by lcm(N, sizeof(T))
  auto APtr2 = malloc_device_annotated<int>(N, q, AL512);
  CHECK_ALIGN(APtr2, 512);
  auto APtr21 = malloc_device_annotated<double>(N, q, AL2);
  CHECK_ALIGN(APtr21, 8);

  // alignment<0>: the allocated raw pointer should be aligned by `sizeof(T)`
  auto APtr22 = malloc_device_annotated<double>(N, q, AL0);
  CHECK_ALIGN(APtr22, 8);

  // alignment arg N1 and property alignment<N2> both specifid:
  // the allocated raw pointer should be aligned by lcm(N1, N2, sizeof(T))
  auto APtr23 =
      aligned_alloc_host_annotated<int>(512 /* alignment */, N, q, AL64);
  CHECK_ALIGN(APtr23, 512);
  auto APtr24 =
      aligned_alloc_host_annotated<int>(64 /* alignment */, N, q, AL512);
  CHECK_ALIGN(APtr24, 512);
  auto APtr25 =
      aligned_alloc_host_annotated<double>(4 /* alignment */, N, q, AL2);
  CHECK_ALIGN(APtr25, 8);

  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    // alignment argument N1 and alignment<N2> property (N1 = 0, N2 > 0),
    // the returned address should be aligned by `lcm(sizeof(T), N2)`
    auto APtr26 = aligned_alloc_shared_annotated<int>(0, N, q, AL8);
    CHECK_ALIGN(APtr26, 8);

    // alignment argument N1 and alignment<N2> property (N1 > 0, N2 = 0),
    // the returned address should be aligned by `lcm(sizeof(T), N1)`
    auto APtr27 = aligned_alloc_shared_annotated<double>(4, N, q, AL0);
    CHECK_ALIGN(APtr27, 8);

    // alignment argument and alignment property are both 0,
    // the returned address should be aligned by `sizeof(T)`
    auto APtr28 = aligned_alloc_shared_annotated<double>(0, N, q, AL0);
    CHECK_ALIGN(APtr28, 8);

    free(APtr26, q);
    free(APtr27, q);
    free(APtr28, q);
  }

  free(APtr1, q);
  free(APtr12, q);
  free(APtr13, q);
  free(APtr14, q);
  free(APtr15, q);
  free(APtr2, q);
  free(APtr21, q);
  free(APtr22, q);
  free(APtr23, q);
  free(APtr24, q);
  free(APtr25, q);
}

int main() {
  sycl::queue q;

  TestUsmKind(q);
  TestAlign(q);
  return 0;
}
