// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %t.out

// UNSUPPORTED: cuda

#include <sycl/sycl.hpp>

#include <cmath>

const static sycl::specialization_id<int> SpecConst1{42};
const static sycl::specialization_id<int> SpecConst2{42};

int main() {
  sycl::queue Q;

  // No support for host device so far
  if (Q.is_host())
    return 0;

  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  KernelBundle.set_specialization_constant<SpecConst1>(1);
  KernelBundle.set_specialization_constant<SpecConst2>(2);

  {
    auto ExecBundle = sycl::build(KernelBundle);
    sycl::buffer<int, 1> Buf{sycl::range{1}};
    Q.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(ExecBundle);
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class Kernel1Name>([=](sycl::kernel_handler KH) {
        Acc[0] = KH.get_specialization_constant<SpecConst1>();
      });
    });
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    assert(Acc[0] == 1);
  }

  {
    auto ExecBundle = sycl::build(KernelBundle);
    sycl::buffer<int, 1> Buf{sycl::range{1}};
    Q.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(ExecBundle);
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class Kernel2Name>([=](sycl::kernel_handler KH) {
        Acc[0] = KH.get_specialization_constant<SpecConst2>();
      });
    });
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    assert(Acc[0] == 2);
  }

  return 0;
}
