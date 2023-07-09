// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Common tests for annotated USM allocation functions based on SYCL
// usm_malloc_properties specification

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/oneapi/usm/usm_alloc.hpp>

#include <iostream>

// clang-format on

using namespace sycl::ext::oneapi::experimental;

properties P1{bar, baz, foo{1}};
properties P2{bar, foo{1}};
properties P3 { bar, baz }
////
//  Test for Device USM allocation functions with properties support
////
void Test(queue &q) {
  // APtr1 is of type annotated_ptr<int, decltype(properties{bar, baz,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr1 = malloc_device_annotated<int>(N, q, P1);

  // APtr2 is of type annotated_ptr<int, decltype(properties{bar,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr2 = malloc_device_annotated<int>(N, q, P2);

  // APtr3 is of type annotated_ptr<int, decltype(properties{bar, baz,
  // usm_kind<sycl::usm::alloc::device>})>
  auto APtr3 = malloc_device_annotated<int>(N, q, P3);

  // Runtime properties are not present on the returned annotated_ptr
  static_assert(std::is_same_v<decltype(APtr1), decltype(APtr3)>);

  // APtr1 and APtr2 do not have the same properties
  static_assert(!std::is_same_v<decltype(APtr1), decltype(APtr2)>);

  // APtr4 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::host>})>
  auto APtr4 = malloc_host_annotated<int>(N, q);

  // APtr5 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::shared>})>
  auto APtr5 = malloc_shared_annotated<int>(N, q);

  // The USM kinds differ
  static_assert(!std::is_same_v<decltype(APtr4), decltype(APtr5)>);

  properties P4{bar, foo{1}};
  // APtr6 is of type annotated_ptr<int, decltype(properties{bar})>
  auto APtr6 = malloc_annotated<int>(N, q, sycl::usm::alloc::device, P4);

  // APtr7 is of type annotated_ptr<int, decltype(properties{})>;
  auto APtr7 = malloc_annotated<int>(N, q, sycl::usm::alloc::device);

  properties P5{usm_kind<sycl::usm::alloc::device>};
  // Throws an exception with error code errc::invalid
  auto APtr8 = malloc_annotated<int>(N, q, sycl::usm::alloc::host, P5);

  // Error: the USM kinds do not agree
  auto APtr9 = malloc_host_annotated<int>(N, q, P5);

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
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr12 is 512-byte
  // aligned
  auto APtr12 = malloc_device_annotated(512, q, P9);

  properties P10{alignment<64>};
  properties P11{alignment<8>};

  // APtr13 is of type annotated_ptr<int,
  // decltype(properties{usm_kind<sycl::usm::alloc::device>})> The raw pointer
  // of APtr13 is 64-byte aligned Note: APtr13 does not have the alignment
  // property. The alignment is runtime information.
  auto APtr13 = aligned_alloc_device_annotated<int>(N, q, 64 /* alignment */);

  // APtr14 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr14 is 64-byte
  // aligned Note: APtr14 has the alignment property because P10 contains the
  // alignment property
  auto APtr14 = aligned_alloc_device_annotated<int>(N, q, 64, P10);

  // APtr15 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr15 is 128-byte
  // aligned Note: APtr15 has the alignment property with value 64, because this
  // is the alignment known at compile-time
  auto APtr15 = aligned_alloc_device_annotated<int>(N, q, 128, P10);

  // APtr16 is of type annotated_ptr<int, decltype(properties{alignment<64>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr16 is 64-byte
  // aligned
  auto APtr16 = aligned_alloc_device_annotated<int>(N, q, 16, P10);

  // APtr17 is of type annotated_ptr<int, decltype(properties{alignment<8>,
  // usm_kind<sycl::usm::alloc::device>})> The raw pointer of APtr17 is 56-byte
  // aligned (if this alignment is supported by the implementation)
  auto APtr17 = aligned_alloc_device_annotated<int>(N, q, 7, P11);
}

////
//  Test for Host USM allocation functions with properties support
////
struct UsmHostIP {
  annotated_ptr<int, MMHostPropListWithUsmHost> a;
  annotated_ptr<void, MMHostPropListWithUsmHost> b;
  annotated_ptr<int, MMHostPropListWithUsmHost> c;
  annotated_ptr<void, MMHostPropListWithUsmHost> d;
  int n;

  void operator()() const {
    for (int i = 0; i < n - 2; i++) {
      a[i + 2] = a[i + 1] + a[i];
    }
    *(a + 1) *= 5;
  }
};

void TestHostAlloc(queue &q) {
  // Create the SYCL device queue
  auto pHost1 =
      malloc_host_annotated<int, MMHostPropList3, MMHostPropListWithUsmHost>(5,
                                                                             q);
  auto pHost2 = malloc_host_annotated<MMHostPropListWithUsmHost,
                                      MMHostPropListWithUsmHost>(5, q);
  auto pHost3 =
      aligned_alloc_host_annotated<int, MMHostPropListWithUsmHost,
                                   MMHostPropListWithUsmHost>(1024, 5, q);
  auto pHost4 =
      aligned_alloc_host_annotated<MMHostPropList3, MMHostPropListWithUsmHost>(
          2048, 5, q);

  for (int i = 0; i < 5; i++) {
    pHost1[i] = i;
    pHost3[i] = i;
    // Arithmetic operations on annotated_ptr<void,...> are not allowed
    // Use `get()` to get the underlying raw pointer and perform the operations
  }

  q.submit([&](handler &h) {
     h.single_task(UsmHostIP{pHost1, pHost2, pHost3, pHost4, 5});
   }).wait();

  free(pHost1, q);
  free(pHost2, q);
  free(pHost3, q);
  free(pHost4, q);
}

////
//  Test for Shared USM allocation functions with properties support
////
struct UsmSharedIP {
  annotated_ptr<int, MMHostPropListWithUsmShared> a;
  annotated_ptr<void, MMHostPropListWithUsmShared> b;
  annotated_ptr<int, MMHostPropListWithUsmShared> c;
  annotated_ptr<void, MMHostPropListWithUsmShared> d;
  int n;

  void operator()() const {
    for (int i = 0; i < n - 2; i++) {
      a[i + 2] = a[i + 1] + a[i];
    }
    *(a + 1) *= 5;
  }
};

void TestSharedAlloc(queue &q) {
  // Create the SYCL device queue
  auto pShared1 = malloc_shared_annotated<int, MMHostPropList2,
                                          MMHostPropListWithUsmShared>(5, q);
  auto pShared2 = malloc_shared_annotated<MMHostPropListWithUsmShared,
                                          MMHostPropListWithUsmShared>(5, q);
  auto pShared3 =
      aligned_alloc_shared_annotated<int, MMHostPropListWithUsmShared,
                                     MMHostPropListWithUsmShared>(1024, 5, q);
  auto pShared4 =
      aligned_alloc_shared_annotated<MMHostPropList2,
                                     MMHostPropListWithUsmShared>(2048, 5, q);

  for (int i = 0; i < 5; i++) {
    pShared1[i] = i;
    pShared3[i] = i;
    // Arithmetic operations on annotated_ptr<void,...> are not allowed
    // Use `get()` to get the underlying raw pointer and perform the operations
  }

  q.submit([&](handler &h) {
     h.single_task(UsmSharedIP{pShared1, pShared2, pShared3, pShared4, 5});
   }).wait();

  free(pShared1, q);
  free(pShared2, q);
  free(pShared3, q);
  free(pShared4, q);
}

int main() {
  queue q(sycl::ext::intel::fpga_selector_v);

  // TestDeviceAlloc(q);
  TestHostAlloc(q);
  TestSharedAlloc(q);
  return 0;
}
