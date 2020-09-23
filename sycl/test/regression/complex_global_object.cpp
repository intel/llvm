// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

class ComplexClass {
public:
  ~ComplexClass() {
    auto Device = sycl::default_selector{}.select_device();
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
