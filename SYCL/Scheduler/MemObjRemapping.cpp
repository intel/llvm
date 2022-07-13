// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia
#include <cassert>
#include <cstddef>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

class Foo;
class Bar;

// This test checks that memory objects are remapped on requesting an access
// mode incompatible with the current mapping. Write access is mapped as
// read-write.
int main() {
  queue Q;

  std::size_t Size = 64;
  range<1> Range{Size};
  buffer<int, 1> BufA{Range};
  buffer<int, 1> BufB{Range};

  Q.submit([&](handler &Cgh) {
    auto AccA = BufA.get_access<access::mode::read_write>(Cgh);
    auto AccB = BufB.get_access<access::mode::read_write>(Cgh);
    Cgh.parallel_for<Foo>(Range, [=](id<1> Idx) {
      AccA[Idx] = Idx[0];
      AccB[Idx] = Idx[0];
    });
  });

  {
    // Check access mode flags
    // CHECK: piEnqueueMemBufferMap
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: : 1
    // CHECK: piEnqueueMemBufferMap
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: : 1
    auto AccA = BufA.get_access<access::mode::read>();
    auto AccB = BufB.get_access<access::mode::read>();
    for (std::size_t I = 0; I < Size; ++I) {
      assert(AccA[I] == I);
      assert(AccB[I] == I);
    }
  }
  {
    // CHECK: piEnqueueMemUnmap
    // CHECK: piEnqueueMemBufferMap
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: : 3
    auto AccA = BufA.get_access<access::mode::write>();
    for (std::size_t I = 0; I < Size; ++I)
      AccA[I] = 2 * I;
  }

  queue HostQ{host_selector()};
  // CHECK: piEnqueueMemUnmap
  // CHECK: piEnqueueMemBufferMap
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: :
  // CHECK-NEXT: : 3
  HostQ.submit([&](handler &Cgh) {
    auto AccB = BufB.get_access<access::mode::write>(Cgh);
    Cgh.parallel_for<Bar>(Range, [=](id<1> Idx) { AccB[Idx] = 2 * Idx[0]; });
  });

  // CHECK-NOT: piEnqueueMemBufferMap
  auto AccA = BufA.get_access<access::mode::read>();
  auto AccB = BufB.get_access<access::mode::read>();
  for (std::size_t I = 0; I < Size; ++I) {
    assert(AccA[I] == 2 * I);
    assert(AccB[I] == 2 * I);
  }
}
