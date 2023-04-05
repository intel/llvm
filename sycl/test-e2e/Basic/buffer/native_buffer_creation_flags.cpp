// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out 2>&1 %CPU_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

class Foo;
using namespace sycl;
int main() {
  const int BufVal = 42;
  buffer<int, 1> Buf{&BufVal, range<1>(1)};
  queue Q;

  {
    // This should trigger memory allocation on host since the pointer passed by
    // the user is read-only.
    auto BufAcc = Buf.get_access<access::mode::write>();
  }

  Q.submit([&](handler &Cgh) {
    // Now that we have a read-write host allocation, check that the native
    // buffer is created with the PI_MEM_FLAGS_HOST_PTR_USE flag.
    // CHECK: piMemBufferCreate
    // CHECK-NEXT: {{.*}} : {{.*}}
    // CHECK-NEXT: {{.*}} : {{.*}}
    // CHECK-NEXT: {{.*}} : 9
    auto BufAcc = Buf.get_access<access::mode::read>(Cgh);
    Cgh.single_task<Foo>([=]() { int A = BufAcc[0]; });
  });
}
