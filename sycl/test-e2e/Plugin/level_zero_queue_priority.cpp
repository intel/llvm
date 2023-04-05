// REQUIRES: gpu, level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZE_DEBUG=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// Check that queue priority is passed to Level Zero runtime
// This is the last value in the ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
//
#include <sycl/sycl.hpp>

void test(sycl::property_list Props) {
  sycl::queue Q(Props);
  (void)Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class EmptyKernel>([=]() {});
  });
  Q.wait();
}

int main(int Argc, const char *Argv[]) {

  // CHECK: [getZeQueue]: create queue {{.*}} priority = Normal
  test(sycl::property_list{});

  // CHECK: [getZeQueue]: create queue {{.*}} priority = Normal
  test({sycl::ext::oneapi::property::queue::priority_normal{}});

  // CHECK: [getZeQueue]: create queue {{.*}} priority = Low
  test({sycl::ext::oneapi::property::queue::priority_low{}});

  // CHECK: [getZeQueue]: create queue {{.*}} priority = High
  test({sycl::ext::oneapi::property::queue::priority_high{}});

  // CHECK: Queue cannot be constructed with different priorities.
  try {
    test({sycl::ext::oneapi::property::queue::priority_low{},
          sycl::ext::oneapi::property::queue::priority_high{}});
  } catch (sycl::exception &E) {
    std::cerr << E.what() << std::endl;
  }

  std::cout << "The test passed." << std::endl;
  return 0;
}
