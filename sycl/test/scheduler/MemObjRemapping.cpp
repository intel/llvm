// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=1 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
#include <CL/sycl.hpp>
#include <cassert>
#include <cstddef>

using namespace cl::sycl;

class Foo;
class Bar;

// This test checks that memory objects are remapped on requesting an access mode
// incompatible with the current mapping.
int main() {
  queue Q;

  std::size_t Size = 1024;
  range<1> Range{Size};
  buffer<int, 1> Buf{Range};

  Q.submit([&](handler &Cgh){
    auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
    Cgh.parallel_for<Foo>(Range, [=](id<1> Idx){
      Acc[Idx] = Idx[0];
    });
  });

  {
    // CHECK: piEnqueueMemBufferMap
    auto Acc = Buf.get_access<access::mode::read>();
    for (std::size_t I = 0; I < Size; ++I)
      assert(Acc[I] == I);
  }
  {
    // CHECK: piEnqueueMemUnmap
    // CHECK: piEnqueueMemBufferMap
    auto Acc = Buf.get_access<access::mode::write>();
    for (std::size_t I = 0; I < Size; ++I)
      Acc[I] = 2 * I;
  }

  queue HostQ{host_selector()};
  // CHECK: piEnqueueMemUnmap
  // CHECK: piEnqueueMemBufferMap
  HostQ.submit([&](handler &Cgh){
    auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
    Cgh.parallel_for<Bar>(Range, [=](id<1> Idx){
      Acc[Idx] *= 2;
    });
  });

  // CHECK-NOT: piEnqueueMemBufferMap
  auto Acc = Buf.get_access<access::mode::read>();
  for (std::size_t I = 0; I < Size; ++I)
    assert(Acc[I] == 4 * I);
}
