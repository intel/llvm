// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Note: Tests that non-trivially copyable types marked as device-copyable are
// copied and used correctly on the device.

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

struct NontriviallyCopyable {
  int i;
  NontriviallyCopyable(int _i) : i(_i) {}
  NontriviallyCopyable(const NontriviallyCopyable &x) : i(x.i) {}
};

template <> struct is_device_copyable<NontriviallyCopyable> : std::true_type {};

int main() {
  queue Q;

  NontriviallyCopyable C(10);

  int Res = 0;
  {
    buffer<int, 1> Buff{&Res, range<1>{1}};

    Q.submit([&](sycl::handler &cgh) {
      auto Acc = Buff.get_access<access::mode::write>(cgh);
      cgh.single_task<class Kernel>([=] { Acc[0] = C.i; });
    });
  }

  if (Res != C.i) {
    std::cout << "Mismatch: " << Res << " != " << C.i << std::endl;
    return 1;
  }

  return 0;
}
