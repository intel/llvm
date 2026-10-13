// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SYCL runtime may construct global objects at function scope. The test ensures
// such objects do not cause problems if the last call to SYCL is in a global
// destructor, provided the SYCL runtime was initialized before that global was
// constructed. Without that, the global's destructor runs after the runtime and
// the backend libraries were torn down at exit, which is not supported.

#include <sycl/detail/core.hpp>

class ComplexClass {
public:
    sycl::device Device;
    ~ComplexClass() {
      sycl::queue Queue{Device};
      sycl::buffer<int, 1> Buf{sycl::range<1>{16}};
      Queue.submit([&](sycl::handler &CGH) {
        auto Acc = Buf.get_access<sycl::access_mode::read_write>(CGH);
        CGH.single_task<class Dummy>([=]() { Acc[0] = 42; });
      });
    }
};

ComplexClass Obj;

int main() {
  ComplexClass Obj2;

  return 0;
}
