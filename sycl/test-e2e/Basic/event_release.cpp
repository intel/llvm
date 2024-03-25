// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

// The test checks that pi_events are released without queue destruction
// or call to queue::wait, when the corresponding commands are cleaned up.

using namespace sycl;

class Foo;

int main() {
  int Val = 0;
  int Gold = 42;

  queue Q;

  {
    buffer<int, 1> Buf{&Val, range<1>(1)};
    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::discard_write>(Cgh);
      Cgh.single_task<Foo>([=]() { Acc[0] = Gold; });
    });
  }

  // Buffer destruction triggers execution graph cleanup, check that both
  // events (one for launching the kernel and one for memory transfer to host)
  // are released.
  // CHECK: piEventRelease
  // CHECK: piEventRelease
  assert(Val == Gold);
  // CHECK: End of main scope
  std::cout << "End of main scope" << std::endl;
}
