// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include "sycl/sycl.hpp"
#include <sycl/ext/oneapi/usm/usm_alloc.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;

using MMHostPropList1 = decltype(properties(awidth<32>));
using MMHostPropListWithUsmDevice = decltype(properties(awidth<32>, usm_kind<sycl::usm::alloc::device>));

using MMHostPropList2 = decltype(properties(buffer_location<1>, alignment<1024>));
using MMHostPropListWithUsmShared = decltype(properties(buffer_location<1>, alignment<1024>, usm_kind<sycl::usm::alloc::shared>));

using MMHostPropList3 = decltype(properties(buffer_location<2>, alignment<1024>));
using MMHostPropListWithUsmHost = decltype(properties(buffer_location<2>, alignment<1024>, usm_kind<sycl::usm::alloc::host>));

////
//  Test for Device USM allocation functions with properties support
////
void TestDeviceAlloc(queue &q) {
  constexpr int kN = 5;
  auto pDevice1 = malloc_device_annotated<int, MMHostPropList1, MMHostPropListWithUsmDevice>(kN, q);
  auto pDevice2 = aligned_alloc_device_annotated<int, MMHostPropList1, MMHostPropListWithUsmDevice>(1024, kN, q);

  int data[kN];
  for (int i = 0; i < kN; i++) {
    data[i] = i;
  }
  q.memcpy(pDevice1.get(), data, kN * sizeof(int));
  q.wait();

  q.parallel_for(kN, [=](id<1> idx) {
    pDevice2[idx] = pDevice1[idx];
  });
  q.wait();

  free(pDevice1, q);
  free(pDevice2, q);
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
  auto pHost1 = malloc_host_annotated<int, MMHostPropList3, MMHostPropListWithUsmHost>(5, q);
  auto pHost2 = malloc_host_annotated<MMHostPropListWithUsmHost, MMHostPropListWithUsmHost>(5, q);
  auto pHost3 = aligned_alloc_host_annotated<int, MMHostPropListWithUsmHost, MMHostPropListWithUsmHost>(1024, 5, q);
  auto pHost4 = aligned_alloc_host_annotated<MMHostPropList3, MMHostPropListWithUsmHost>(2048, 5, q);

  for (int i = 0; i < 5; i++) {
    pHost1[i] = i;
    pHost3[i] = i;
    // Arithmetic operations on annotated_ptr<void,...> are not allowed
    // Use `get()` to get the underlying raw pointer and perform the operations
  }

  q.submit([&](handler &h) { h.single_task(UsmHostIP{pHost1, pHost2, pHost3, pHost4, 5}); }).wait();

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
  auto pShared1 = malloc_shared_annotated<int, MMHostPropList2, MMHostPropListWithUsmShared>(5, q);
  auto pShared2 = malloc_shared_annotated<MMHostPropListWithUsmShared, MMHostPropListWithUsmShared>(5, q);
  auto pShared3 = aligned_alloc_shared_annotated<int, MMHostPropListWithUsmShared, MMHostPropListWithUsmShared>(1024, 5, q);
  auto pShared4 = aligned_alloc_shared_annotated<MMHostPropList2, MMHostPropListWithUsmShared>(2048, 5, q);

  for (int i = 0; i < 5; i++) {
    pShared1[i] = i;
    pShared3[i] = i;
    // Arithmetic operations on annotated_ptr<void,...> are not allowed
    // Use `get()` to get the underlying raw pointer and perform the operations
  }

  q.submit([&](handler &h) { h.single_task(UsmSharedIP{pShared1, pShared2, pShared3, pShared4, 5}); }).wait();

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
