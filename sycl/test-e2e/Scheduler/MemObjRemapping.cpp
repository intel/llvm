// RUN: %{build} -o %t.out
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
//
// XFAIL: hip_nvidia
#include <cassert>
#include <cstddef>
#include <sycl/detail/core.hpp>

using namespace sycl;

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

  Q.submit([&](handler &Cgh) {
    auto AccA = BufA.get_access<access::mode::read_write>(Cgh);
    Cgh.parallel_for<Foo>(Range, [=](id<1> Idx) { AccA[Idx] = Idx[0]; });
  });

  {
    // Check access mode flags
    // CHECK: piEnqueueMemBufferMap
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: : 1
    host_accessor AccA(BufA, read_only);
    for (std::size_t I = 0; I < Size; ++I) {
      assert(AccA[I] == I);
    }
  }
  {
    // CHECK: piEnqueueMemUnmap
    // CHECK: piEnqueueMemBufferMap
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: :
    // CHECK-NEXT: : 3
    host_accessor AccA(BufA, write_only);
    for (std::size_t I = 0; I < Size; ++I)
      AccA[I] = 2 * I;
  }

  // CHECK-NOT: piEnqueueMemBufferMap
  host_accessor AccA(BufA, read_only);
  for (std::size_t I = 0; I < Size; ++I) {
    assert(AccA[I] == 2 * I);
  }
}
