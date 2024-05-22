// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// XFAIL: gpu && linux

// SYCL runtime may construct global objects at function scope. The test ensures
// such objects do not cause problems if the first call to SYCL is inside main
// and the last call is in global destructor.

// REQUIRES: TEMPORARY_DISABLED
// Disable test due to flacky failures

#include <sycl/detail/core.hpp>

class ComplexClass {
public:
  ~ComplexClass() {
    sycl::device Device;
    sycl::queue Queue{Device};
    sycl::buffer<int, 1> Buf{sycl::range<1>{16}};
    Queue.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class Dummy>([=]() { Acc[0] = 42; });
    });
  }
};

ComplexClass Obj;

int main() {
  ComplexClass Obj2;

  return 0;
}
