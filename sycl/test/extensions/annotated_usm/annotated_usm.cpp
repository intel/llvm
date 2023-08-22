// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Common tests for annotated USM allocation

#include "sycl/sycl.hpp"

#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

constexpr int N = 10;

void TestAlloc(sycl::queue &q) {
  // P1: input properties
  properties P1{conduit, stable, cache_config{large_slm}};
  // ANNP1: output properties with usm_kind and no runtime property
  // `cache_config`
  properties ANNP1{conduit, stable, usm_kind<alloc::device>};
  auto APtr1 = malloc_device_annotated<int>(N, q, P1);
  auto APtr11 = malloc_device_annotated<int, decltype(P1)>(N, q, P1);
  auto APtr12 =
      malloc_device_annotated<int, decltype(P1), decltype(ANNP1)>(N, q, P1);
  // Check that the annotated_ptr allocated should contain properties in ANNP1
  static_assert(
      std::is_same_v<decltype(APtr1), annotated_ptr<int, decltype(ANNP1)>>);
  static_assert(std::is_same_v<decltype(APtr11), decltype(APtr11)>);
  static_assert(std::is_same_v<decltype(APtr12), decltype(APtr12)>);

  // P2: input properties
  properties P2{conduit, buffer_location<5>};
  // ANNP2: output properties with usm_kind
  properties ANNP2{conduit, buffer_location<5>, usm_kind<alloc::device>};
  auto APtr2 = malloc_device_annotated<int>(N, q, P2);
  static_assert(
      std::is_same_v<decltype(APtr2), annotated_ptr<int, decltype(ANNP2)>>);

  // P3: input properties
  properties P3{conduit, stable, cache_config{large_data}};
  // ANNP3: output properties with usm_kind and no runtime property
  // `cache_config`
  properties ANNP3{conduit, stable, usm_kind<alloc::device>};
  auto APtr3 = malloc_device_annotated<int>(N, q, P3);
  static_assert(
      std::is_same_v<decltype(APtr3), annotated_ptr<int, decltype(ANNP3)>>);

  // APtr4 is of type annotated_ptr<void, decltype(properties{conduit, stable,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr4 = malloc_device_annotated(N, q, P2);
  auto APtr41 = malloc_device_annotated<decltype(P2)>(N, q);
  auto APtr42 = malloc_device_annotated<decltype(P2), decltype(ANNP2)>(N, q);
  static_assert(
      std::is_same_v<decltype(APtr4), annotated_ptr<void, decltype(ANNP2)>>);
  static_assert(std::is_same_v<decltype(APtr4), decltype(APtr41)>);
  static_assert(std::is_same_v<decltype(APtr4), decltype(APtr42)>);

  // APtr3 and APtr4 are not same because the underlying pointer type is int* vs
  // void*
  static_assert(!std::is_same_v<decltype(APtr4), decltype(APtr3)>);

  // APtr1 and APtr2 do not have the same properties
  static_assert(!std::is_same_v<decltype(APtr1), decltype(APtr2)>);

  auto APtr5 = malloc_device_annotated<int>(N, q);
  auto APtr51 = aligned_alloc_device_annotated<int>(128, N, q);
  static_assert(
      std::is_same_v<
          decltype(APtr5),
          annotated_ptr<int, decltype(properties{usm_kind<alloc::device>})>>);
  static_assert(std::is_same_v<decltype(APtr5), decltype(APtr51)>);

  auto APtr52 = malloc_device_annotated(N, q);
  auto APtr53 = aligned_alloc_device_annotated(16, N, q);
  static_assert(
      std::is_same_v<
          decltype(APtr52),
          annotated_ptr<void, decltype(properties{usm_kind<alloc::device>})>>);
  static_assert(std::is_same_v<decltype(APtr52), decltype(APtr53)>);

  auto APtr6 = malloc_host_annotated<int>(N, q);
  static_assert(
      std::is_same_v<
          decltype(APtr6),
          annotated_ptr<int, decltype(properties{usm_kind<alloc::host>})>>);

  auto APtr7 = malloc_shared_annotated<int>(N, q);
  static_assert(
      std::is_same_v<
          decltype(APtr7),
          annotated_ptr<int, decltype(properties{usm_kind<alloc::shared>})>>);

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
  // P4: input properties
  properties P4{conduit, cache_config{large_slm}};
  // ANNP4: output properties, with no usm_kind and no runtime property
  properties ANNP4{conduit};
  auto APtr1 = malloc_annotated<int>(N, q, alloc::device, P4);
  static_assert(
      std::is_same_v<decltype(APtr1), annotated_ptr<int, decltype(ANNP4)>>);

  // APtr7 is of type annotated_ptr<int, decltype(properties{})>;
  auto APtr2 = malloc_annotated<int>(N, q, alloc::device);
  static_assert(std::is_same_v<decltype(APtr2), annotated_ptr<int>>);

  properties P5{usm_kind<alloc::host>};
  // Throws an exception with error code errc::invalid
  // auto APtr8 = malloc_annotated<int>(N, q, sycl::usm::alloc::device, P5);

  // Error: the USM kinds do not agree
  // auto APtr9 = malloc_shared_annotated<int>(N, q, P5);

  // the USM kind is required in the propList
  auto APtr3 = malloc_annotated<int>(N, q, P5);
  static_assert(
      std::is_same_v<decltype(APtr3), annotated_ptr<int, decltype(P5)>>);

  auto APtr31 = malloc_annotated(N, q, P5);
  static_assert(
      std::is_same_v<decltype(APtr31), annotated_ptr<void, decltype(P5)>>);

  free(APtr1, q);
  free(APtr2, q);
  free(APtr3, q);
  free(APtr31, q);
}

void TestAlign(sycl::queue &q) {
  // P7: input properties
  properties P7{alignment<512>};
  // ANNP7: output properties with usm_kind
  properties ANNP7{alignment<512>, usm_kind<alloc::device>};
  // The raw pointer of APtr1 is 512-byte aligned
  auto APtr1 = malloc_device_annotated<int>(N, q, P7);
  static_assert(
      std::is_same_v<decltype(APtr1), annotated_ptr<int, decltype(ANNP7)>>);

  // P8: input properties
  properties P8{alignment<2>};
  // ANNP8: output properties with usm_kind
  properties ANNP8{alignment<2>, usm_kind<alloc::device>};
  // The raw pointer of APtr2 is sizeof(int)-byte aligned, e.g., 4 for some
  // implementations
  auto APtr2 = malloc_device_annotated<int>(N, q, P8);
  static_assert(
      std::is_same_v<decltype(APtr2), annotated_ptr<int, decltype(ANNP8)>>);

  // P9: input properties
  properties P9{alignment<64>};
  // ANNP9: output properties with usm_kind
  properties ANNP9{alignment<64>, usm_kind<alloc::device>};
  // The raw pointer of APtr3 is 64-byte aligned
  auto APtr3 = malloc_device_annotated(N, q, P9);
  static_assert(
      std::is_same_v<decltype(APtr3), annotated_ptr<void, decltype(ANNP9)>>);

  // The raw pointer of APtr4 is 1023-byte aligned
  // Note: APtr4 does not have the alignment
  // property. The alignment is runtime information.
  auto APtr4 = aligned_alloc_device_annotated<int>(
      1024 /* alignment */, N, q,
      sycl::ext::oneapi::experimental::detail::empty_properties_t{});
  static_assert(
      std::is_same_v<
          decltype(APtr4),
          annotated_ptr<int, decltype(properties{usm_kind<alloc::device>})>>);

  auto APtr41 = aligned_alloc_device_annotated(1024 /* alignment */, N, q);
  static_assert(
      std::is_same_v<
          decltype(APtr41),
          annotated_ptr<void, decltype(properties{usm_kind<alloc::device>})>>);

  // The raw pointer of APtr5 is 64-byte aligned
  // Note: APtr5 has the alignment property because P9 contains the
  // alignment property
  auto APtr5 = aligned_alloc_device_annotated<int>(64, N, q, P9);
  static_assert(
      std::is_same_v<decltype(APtr5), annotated_ptr<int, decltype(ANNP9)>>);

  // The raw pointer of APtr6 is 128-byte
  // aligned Note: APtr15 has the alignment property with value 64, because this
  // is the alignment known at compile-time
  auto APtr6 = aligned_alloc_device_annotated<int>(128, N, q, P9);
  static_assert(
      std::is_same_v<decltype(APtr6), annotated_ptr<int, decltype(ANNP9)>>);

  // The raw pointer of APtr7 is 64-byte aligned
  auto APtr7 = aligned_alloc_device_annotated<int>(16, N, q, P9);
  static_assert(
      std::is_same_v<decltype(APtr7), annotated_ptr<int, decltype(ANNP9)>>);

  // P10: input properties
  properties P10{alignment<8>};
  // ANNP10: output properties with usm_kind
  properties ANNP10{alignment<8>, usm_kind<alloc::device>};
  // The raw pointer of APtr8 is 56-byte
  // aligned (if this alignment is supported by the implementation)
  auto APtr8 = aligned_alloc_device_annotated<int>(7, N, q, P10);
  static_assert(
      std::is_same_v<decltype(APtr8), annotated_ptr<int, decltype(ANNP10)>>);

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

  TestAlloc(q);
  TestProperty(q);
  TestAlign(q);
  return 0;
}
