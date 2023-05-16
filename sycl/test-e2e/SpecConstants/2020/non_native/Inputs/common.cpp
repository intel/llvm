#include <sycl/sycl.hpp>

#include <cmath>

class Kernel1Name;
class Kernel2Name;

struct TestStruct {
  int a;
  int b;
};

const static sycl::specialization_id<int> SpecConst1{42};
const static sycl::specialization_id<int> SpecConst2{42};
const static sycl::specialization_id<TestStruct> SpecConst3{TestStruct{42, 42}};
const static sycl::specialization_id<short> SpecConst4{42};

int main() {
  sycl::queue Q;

  {
    sycl::buffer<int, 1> Buf{sycl::range{1}};
    Q.submit([&](sycl::handler &CGH) {
      CGH.set_specialization_constant<SpecConst2>(1);
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class Kernel1Name>([=](sycl::kernel_handler KH) {
        Acc[0] = KH.get_specialization_constant<SpecConst2>();
      });
    });
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    assert(Acc[0] == 1);
  }

  {
    sycl::buffer<TestStruct, 1> Buf{sycl::range{1}};
    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.set_specialization_constant<SpecConst3>(TestStruct{1, 2});
      const auto SC = CGH.get_specialization_constant<SpecConst4>();
      assert(SC == 42);
      CGH.single_task<class Kernel2Name>([=](sycl::kernel_handler KH) {
        Acc[0] = KH.get_specialization_constant<SpecConst3>();
      });
    });
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    assert(Acc[0].a == 1 && Acc[0].b == 2);
  }

  return 0;
}
