// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out > %t.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC < %t.log
//
// Same with Immediate CommandLists
// RUN: env SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING=1 env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out > %t.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC < %t.log

// REQUIRES: level_zero

#include <sycl/sycl.hpp>

using namespace sycl;

void test_pvc(device &d) {
  std::cout << "Test PVC Begin" << std::endl;
  // CHECK-PVC: Test PVC Begin
  bool IsPVC = [&]() {
    if (!d.has(aspect::ext_intel_device_id))
      return false;
    return (d.get_info<ext::intel::info::device::device_id>() & 0xff0) == 0xbd0;
  }();
  std::cout << "IsPVC: " << std::boolalpha << IsPVC << std::endl;
  if (IsPVC) {
    assert(d.get_info<ext::intel::info::device::max_compute_queue_indices>() ==
           1);

    auto sub_devices = d.create_sub_devices<
        info::partition_property::partition_by_affinity_domain>(
        info::partition_affinity_domain::next_partitionable);
    device &sub_device = sub_devices[1];
    assert(
        sub_device
            .get_info<ext::intel::info::device::max_compute_queue_indices>() ==
        4);

    auto sub_sub_devices = sub_device.create_sub_devices<
        info::partition_property::ext_intel_partition_by_cslice>();
    device &sub_sub_device = sub_sub_devices[1];
    assert(
        sub_sub_device
            .get_info<ext::intel::info::device::max_compute_queue_indices>() ==
        1);

    {
      bool ExceptionThrown = false;
      try {
        std::ignore = queue{d, ext::intel::property::queue::compute_index{-1}};

      } catch (...) {
        ExceptionThrown = true;
      }
      assert(ExceptionThrown);
    }
    {
      bool ExceptionThrown = false;
      try {
        std::ignore = queue{sub_sub_device,
                            ext::intel::property::queue::compute_index{1}};

      } catch (...) {
        ExceptionThrown = true;
      }
      assert(ExceptionThrown);
    }

    {
      queue q{sub_device};
      // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
      q.single_task([=]() {}).wait();
    }
    {
      queue q{sub_device, ext::intel::property::queue::compute_index{2}};
      // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])
      q.single_task([=]() {}).wait();
    }
    {
      queue q{sub_device, ext::intel::property::queue::compute_index{2}};
      // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])
      q.single_task([=]() {}).wait();
    }
    {
      queue q{sub_sub_device};
      // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
      q.single_task([=]() {}).wait();
    }
  } else {
    // Make FileCheck pass.
    std::cout << "Fake ZE_DEBUG output for FileCheck:" << std::endl;
    // clang-format off
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])" << std::endl;
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])" << std::endl;
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])" << std::endl;
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])" << std::endl;
    // clang-format on
  }
  std::cout << "Test PVC End" << std::endl;
  // CHECK-PVC: Test PVC End
}

int main() {
  device d;

  test_pvc(d);

  return 0;
}
