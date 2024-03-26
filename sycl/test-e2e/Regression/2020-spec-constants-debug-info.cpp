// RUN: %{build} -g -o %t.out
// RUN: %{build} -g -O0 -o %t.out
// RUN: %{build} -g -O2 -o %t.out
//
// The idea of this test is to make sure that we can compile the following
// simple example without crashes/assertions firing at llvm-spirv step due to
// debug info corrupted by sycl-post-link

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

constexpr sycl::specialization_id<int> test_id_1{42};

int main() {

  sycl::queue q;
  {
    sycl::buffer<double, 1> Buf{sycl::range{1}};
    q.submit([&](sycl::handler &cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.set_specialization_constant<test_id_1>(1);
      cgh.single_task<class Kernel1>([=](sycl::kernel_handler kh) {
        Acc[0] = kh.get_specialization_constant<test_id_1>();
      });
    });
    auto Acc = Buf.get_host_access();
    assert(Acc[0] == 1);
  }
  return 0;
}
