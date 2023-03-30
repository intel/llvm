// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: %BE_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: hip

#include <sycl/sycl.hpp>

#include <cmath>

class Kernel1Name;
class Kernel2Name;

struct TestStruct {
  int a;
  int b;
};

struct TestStruct2 {
  bool a;
  long b;
};

const static sycl::specialization_id<int> SpecConst1{42};
const static sycl::specialization_id<int> SpecConst2{42};
const static sycl::specialization_id<TestStruct> SpecConst3{TestStruct{42, 42}};
const static sycl::specialization_id<short> SpecConst4{42};
const static sycl::specialization_id<TestStruct2> SpecConst5{
    TestStruct2{true, 0}};

int main() {
  sycl::queue Q;

  // The code is needed to just have device images in the executable
  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel1Name>([] {}); });
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([] {}); });
  }

  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  assert(KernelBundle.contains_specialization_constants() == true);
  assert(KernelBundle.has_specialization_constant<SpecConst1>() == false);
  assert(KernelBundle.has_specialization_constant<SpecConst2>() == true);

  // Test that unused spec constants are saved.
  assert(KernelBundle.get_specialization_constant<SpecConst1>() == 42);
  KernelBundle.set_specialization_constant<SpecConst1>(1);
  assert(KernelBundle.get_specialization_constant<SpecConst1>() == 1);

  KernelBundle.set_specialization_constant<SpecConst2>(1);
  {
    auto ExecBundle = sycl::build(KernelBundle);
    assert(ExecBundle.get_specialization_constant<SpecConst1>() == 1);
    assert(ExecBundle.get_specialization_constant<SpecConst2>() == 1);
    sycl::buffer<int, 1> Buf{sycl::range{1}};
    Q.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(ExecBundle);
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.single_task<class Kernel3Name>([=](sycl::kernel_handler KH) {
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
      CGH.single_task<class Kernel4Name>([=](sycl::kernel_handler KH) {
        Acc[0] = KH.get_specialization_constant<SpecConst3>();
      });
    });
    auto Acc = Buf.get_access<sycl::access::mode::read>();
    assert(Acc[0].a == 1 && Acc[0].b == 2);
  }

  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  KernelBundle2.set_specialization_constant<SpecConst5>(TestStruct2{false, 1});

  sycl::kernel_bundle KernelBundle3 =
      sycl::join<sycl::bundle_state::input>({KernelBundle2, KernelBundle});

  assert(KernelBundle3.get_specialization_constant<SpecConst5>().a == false);
  assert(KernelBundle3.get_specialization_constant<SpecConst5>().b == 1);
  assert(KernelBundle3.get_specialization_constant<SpecConst1>() == 1);
  assert(KernelBundle3.get_specialization_constant<SpecConst2>() == 1);

  return 0;
}
