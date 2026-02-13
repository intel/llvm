// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

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
    // buffer is created with the UR_MEM_FLAG_USE_HOST_POINTER flag.
    // CHECK: <--- urMemBufferCreate
    // CHECK-SAME: UR_MEM_FLAG_USE_HOST_POINTER
    auto BufAcc = Buf.get_access<access::mode::read>(Cgh);
    Cgh.single_task<Foo>([=]() { int A = BufAcc[0]; });
  });
}
