// REQUIRES: gpu, level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-STD
// RUN: env UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-IMM
//
// Check that queue priority is passed to Level Zero runtime
// This is the last value in the ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
//
// With immediate command lists the command lists are recycled between queues in
// a context.
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

void test(sycl::context &C, sycl::device &D, sycl::property_list Props) {
  sycl::queue Q(C, D, Props);
  (void)Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class EmptyKernel>([=]() {});
  });
  Q.wait();
}

int main(int Argc, const char *Argv[]) {
  sycl::device D;
  sycl::context C{D};

  // CHECK-STD: [getZeQueue]: create queue {{.*}} priority = Normal
  // CHECK-IMM: [getZeQueue]: create queue {{.*}} priority = Normal
  test(C, D, sycl::property_list{});

  // CHECK-STD: [getZeQueue]: create queue {{.*}} priority = Normal
  // With immediate command list recycling, a new IMM is not created here.
  // CHECK-IMM-NOT: [getZeQueue]: create queue {{.*}} priority = Normal
  test(C, D, {sycl::ext::oneapi::property::queue::priority_normal{}});

  // CHECK-STD: [getZeQueue]: create queue {{.*}} priority = Low
  // CHECK-IMM: [getZeQueue]: create queue {{.*}} priority = Low
  test(C, D, {sycl::ext::oneapi::property::queue::priority_low{}});

  // CHECK-STD: [getZeQueue]: create queue {{.*}} priority = High
  // CHECK-IMM: [getZeQueue]: create queue {{.*}} priority = High
  test(C, D, {sycl::ext::oneapi::property::queue::priority_high{}});

  // CHECK-STD: Queue cannot be constructed with different priorities.
  // CHECK-IMM: Queue cannot be constructed with different priorities.
  try {
    test(C, D,
         {sycl::ext::oneapi::property::queue::priority_low{},
          sycl::ext::oneapi::property::queue::priority_high{}});
  } catch (sycl::exception &E) {
    std::cerr << E.what() << std::endl;
  }

  std::cout << "The test passed." << std::endl;
  return 0;
}
