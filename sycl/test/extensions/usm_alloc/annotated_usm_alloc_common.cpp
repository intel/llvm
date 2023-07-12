// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Common tests for annotated USM allocation functions based on SYCL
// usm_malloc_properties specification

#include "sycl/sycl.hpp"

#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestAlloc(sycl::queue &q) {
  properties P1{conduit, stable, cache_config{large_slm}};
  properties P2{conduit, buffer_location<5>};
  properties P3{conduit, stable, cache_config{large_data}};
  properties ANNP1{conduit, stable, usm_kind<alloc::device>};
  properties ANNP2{conduit, buffer_location<5>, usm_kind<alloc::device>};

  // APtr1 is of type annotated_ptr<int, decltype(properties{conduit, stable,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr1 = malloc_device_annotated<int>(N, q, P1);
  auto APtr11 = malloc_device_annotated<int, decltype(P1)>(N, q, P1);
  auto APtr12 =
      malloc_device_annotated<int, decltype(P1), decltype(ANNP1)>(N, q, P1);

  // APtr2 is of type annotated_ptr<int, decltype(properties{conduit,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr2 = malloc_device_annotated<int>(N, q, P2);

  // APtr3 is of type annotated_ptr<int, decltype(properties{conduit, stable,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr3 = malloc_device_annotated<int>(N, q, P3);

  // APtr4 is of type annotated_ptr<void, decltype(properties{conduit, stable,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr4 = malloc_device_annotated(N, q, P2);
  auto APtr41 = malloc_device_annotated<decltype(P2)>(N, q);
  auto APtr42 = malloc_device_annotated<decltype(P2), decltype(ANNP2)>(N, q);

  // APtr3 and APtr4 are not same because the underlying pointer type is int* vs
  // void*
  static_assert(!std::is_same_v<decltype(APtr4), decltype(APtr3)>);

  // Runtime properties are not present on the returned annotated_ptr
  static_assert(std::is_same_v<decltype(APtr1), decltype(APtr3)>);

  // APtr1 and APtr2 do not have the same properties
  static_assert(!std::is_same_v<decltype(APtr1), decltype(APtr2)>);

  auto APtr5 = malloc_device_annotated<int>(N, q);
  auto APtr51 = malloc_device_annotated(N, q);
  auto APtr52 = aligned_alloc_device_annotated(16, N, q);
  auto APtr53 = aligned_alloc_device_annotated<int>(128, N, q);

  static_assert(std::is_same_v<decltype(APtr5), decltype(APtr53)>);
  static_assert(
      std::is_same_v<
          decltype(APtr5),
          annotated_ptr<int, decltype(properties{usm_kind<alloc::device>})>>);
  static_assert(std::is_same_v<decltype(APtr51), decltype(APtr52)>);
  static_assert(
      std::is_same_v<
          decltype(APtr51),
          annotated_ptr<void, decltype(properties{usm_kind<alloc::device>})>>);

  // APtr6 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::host>})>
  auto APtr6 = malloc_host_annotated<int>(N, q);

  // APtr7 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::shared>})>
  auto APtr7 = malloc_shared_annotated<int>(N, q);

  // The USM kinds differ
  static_assert(!std::is_same_v<decltype(APtr6), decltype(APtr7)>);

  free(APtr1, q);
  free(APtr11, q);
  free(APtr12, q);
  free(APtr2, q);
  free(APtr3, q);
  free(APtr4, q);
  free(APtr41, q);
  free(APtr42, q);
  free(APtr5, q);
  free(APtr51, q);
  free(APtr52, q);
  free(APtr53, q);
  free(APtr6, q);
  free(APtr7, q);
}

void TestProperty(sycl::queue &q) {
  properties P4{conduit, cache_config{large_slm}};
  // APtr6 is of type annotated_ptr<int, decltype(properties{conduit})>
  auto APtr6 = malloc_annotated<int>(N, q, sycl::usm::alloc::device, P4);

  // APtr7 is of type annotated_ptr<int, decltype(properties{})>;
  auto APtr7 = malloc_annotated<int>(N, q, sycl::usm::alloc::device);

  properties P5{usm_kind<sycl::usm::alloc::host>};
  // Throws an exception with error code errc::invalid
  // auto APtr8 = malloc_annotated<int>(N, q, sycl::usm::alloc::device, P5);

  // Error: the USM kinds do not agree
  // auto APtr9 = malloc_shared_annotated<int>(N, q, P5);

  // the USM kind is required in the propList
  auto APtr10 = malloc_annotated<int>(N, q, P5);
  auto APtr101 = malloc_annotated(N, q, P5);

  free(APtr6, q);
  free(APtr7, q);
  free(APtr10, q);
  free(APtr101, q);
}

void TestAlign(sycl::queue &q) {
  properties P7{alignment<512>};
  properties P8{alignment<2>};
  properties P9{alignment<64>};

  // APtr10 is of type annotated_ptr<int, decltype(properties{alignment<512>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr10 is 512-byte
  // aligned
  auto APtr10 = malloc_device_annotated<int>(N, q, P7);

  // APtr11 is of type annotated_ptr<int, decltype(properties{alignment<1>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr11 is
  // sizeof(int)-byte aligned, e.g., 4 for some implementations
  auto APtr11 = malloc_device_annotated<int>(N, q, P8);

  // APtr12 is of type annotated_ptr<void, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr12 is 64-byte
  // aligned
  auto APtr12 = malloc_device_annotated(N, q, P9);

  properties P10{alignment<64>};
  properties P11{alignment<8>};

  // APtr13 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::device>})> The raw pointer
  // of APtr13 is 64-byte aligned Note: APtr13 does not have the alignment
  // property. The alignment is runtime information.
  auto APtr13 = aligned_alloc_device_annotated<int>(
      1024 /* alignment */, N, q,
      sycl::ext::oneapi::experimental::detail::empty_properties_t{});

  auto APtr131 = aligned_alloc_device_annotated(1024 /* alignment */, N, q);

  // APtr14 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr14 is 64-byte
  // aligned Note: APtr14 has the alignment property because P10 contains the
  // alignment property
  auto APtr14 = aligned_alloc_device_annotated<int>(64, N, q, P10);

  // APtr15 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr15 is 128-byte
  // aligned Note: APtr15 has the alignment property with value 64, because this
  // is the alignment known at compile-time
  auto APtr15 = aligned_alloc_device_annotated<int>(128, N, q, P10);

  // APtr16 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr16 is 64-byte
  // aligned
  auto APtr16 = aligned_alloc_device_annotated<int>(16, N, q, P10);

  // APtr17 is of type annotated_ptr<int, decltype(properties{alignment<8>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr17 is 56-byte
  // aligned (if this alignment is supported by the implementation)
  auto APtr17 = aligned_alloc_device_annotated<int>(7, N, q, P11);

  free(APtr10, q);
  free(APtr11, q);
  free(APtr12, q);
  free(APtr13, q);
  free(APtr131, q);
  free(APtr14, q);
  free(APtr15, q);
  free(APtr16, q);
  free(APtr17, q);
}

int main() {
  sycl::queue q;

  TestAlloc(q);
  TestProperty(q);
  TestAlign(q);
  return 0;
}
