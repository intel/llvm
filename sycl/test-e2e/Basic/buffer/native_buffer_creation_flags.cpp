// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

class Foo;
using namespace sycl;
int main() {
  const int BufVal = 42;
  buffer<int, 1> Buf{&BufVal, range<1>(1)};
  queue Q;

  {
    // This should trigger memory allocation on host since the pointer passed by
    // the user is read-only.
    host_accessor BufAcc(Buf, write_only);
  }

  Q.submit([&](handler &Cgh) {
    // Now that we have a read-write host allocation, check that the native
    // buffer is created with the PI_MEM_FLAGS_HOST_PTR_USE flag.
    // CHECK: piMemBufferCreate
    // CHECK-NEXT: {{.*}} : {{.*}}
    // CHECK-NEXT: {{.*}} : 9
    auto BufAcc = Buf.get_access<access::mode::read>(Cgh);
    Cgh.single_task<Foo>([=]() { int A = BufAcc[0]; });
  });
}
