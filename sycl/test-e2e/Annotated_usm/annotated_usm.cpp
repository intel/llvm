// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// E2E test for annotated USM allocation

#include "sycl/sycl.hpp"
#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  // the USM kind is specified in the function name
  properties P1{conduit, stable};
  auto APtr1 = malloc_device_annotated<int>(N, q, P1);
  assert(sycl::get_pointer_type(APtr1.get(), Ctx) == alloc::device);

  auto APtr2 = aligned_alloc_device_annotated(128, N, q);
  assert(sycl::get_pointer_type(APtr2.get(), Ctx) == alloc::device);

  auto APtr3 = malloc_host_annotated<int>(N, q);
  assert(sycl::get_pointer_type(APtr3.get(), Ctx) == alloc::host);

  if (dev.has(sycl::aspect::usm_shared_allocations)) {
    auto APtr4 = malloc_shared_annotated(N, q);
    assert(sycl::get_pointer_type(APtr4.get(), Ctx) == alloc::shared);
    free(APtr4, q);
  }

  // the USM kind is specified in the propList
  properties P2{conduit, cache_config{large_slm}};
  auto APtr5 = malloc_annotated<int>(N, q, alloc::device, P2);
  assert(sycl::get_pointer_type(APtr5.get(), Ctx) == alloc::device);

  // the USM kind is specified in the propList
  properties P3{usm_kind<alloc::host>};
  auto APtr6 = malloc_annotated<int>(N, q, P3);
  assert(sycl::get_pointer_type(APtr6.get(), Ctx) == alloc::host);

  free(APtr1, q);
  free(APtr2, q);
  free(APtr3, q);
  free(APtr5, q);
  free(APtr6, q);
}

void TestAlign(sycl::queue &q) {
  properties P7{alignment<512>};
  // The raw pointer of APtr1 is 512-byte aligned
  auto APtr1 = malloc_device_annotated<int>(N, q, P7);
  assert(((uintptr_t)APtr1.get() & 511) == 0);

  properties P8{alignment<2>};
  // The raw pointer of APtr2 is sizeof(double)-byte aligned, e.g., 8 for some
  // implementations
  auto APtr2 = malloc_device_annotated<double>(N, q, P8);
  assert(((uintptr_t)APtr2.get() & 7) == 0);

  properties P9{alignment<64>};
  // The raw pointer of APtr3 is 64-byte aligned
  auto APtr3 = malloc_device_annotated(N, q, P9);
  assert(((uintptr_t)APtr3.get() & 63) == 0);

  // The raw pointer of APtr4 is 1024-byte aligned
  // Note: APtr4 does not have the alignment
  // property. The alignment is runtime information.
  auto APtr4 = aligned_alloc_device_annotated<int>(
      1024 /* alignment */, N, q,
      sycl::ext::oneapi::experimental::detail::empty_properties_t{});
  assert(((uintptr_t)APtr4.get() & 1023) == 0);

  auto APtr41 = aligned_alloc_device_annotated(1024 /* alignment */, N, q);
  assert(((uintptr_t)APtr41.get() & 1023) == 0);

  // The raw pointer of APtr5 is 64-byte aligned
  auto APtr5 = aligned_alloc_device_annotated<int>(64, N, q, P9);
  assert(((uintptr_t)APtr5.get() & 63) == 0);

  // The raw pointer of APtr6 is 128-byte
  // aligned Note: APtr15 has the alignment property with value 64, because this
  // is the alignment known at compile-time
  auto APtr6 = aligned_alloc_device_annotated<int>(128, N, q, P9);
  assert(((uintptr_t)APtr6.get() & 127) == 0);

  // The raw pointer of APtr7 is 64-byte aligned
  auto APtr7 = aligned_alloc_device_annotated<int>(16, N, q, P9);
  assert(((uintptr_t)APtr7.get() & 63) == 0);

  properties P10{alignment<8>};
  // The raw pointer of APtr8 is 56-byte
  // aligned (if this alignment is supported by the implementation)
  auto APtr8 = aligned_alloc_device_annotated<int>(7, N, q, P10);
  assert(((uintptr_t)APtr8.get() % 56) == 0);

  free(APtr1, q);
  free(APtr2, q);
  free(APtr3, q);
  free(APtr4, q);
  free(APtr41, q);
  free(APtr5, q);
  free(APtr6, q);
  free(APtr7, q);
  free(APtr8, q);
}

int main() {
  sycl::queue q;

  TestUsmKind(q);
  TestAlign(q);
  return 0;
}
