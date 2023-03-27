// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out

// RUN: env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out > %t.default.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC < %t.default.log

// RUN: env SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING=1 \
// RUN:   env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out> %t.compat.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC,CHECK-PVC-AFFINITY < %t.compat.log

// Same, but using immediate commandlists:

// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out > %t.default.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC < %t.default.log

// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 env SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING=1 \
// RUN:   env ZEX_NUMBER_OF_CCS=0:4 env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out> %t.compat.log 2>&1
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --check-prefixes=CHECK-PVC,CHECK-PVC-AFFINITY < %t.compat.log

// REQUIRES: level_zero

#include <sycl/sycl.hpp>

using namespace sycl;

// Specified in the RUN line.
static constexpr int NumCSlices = 4;
static const bool ExposeCSliceInAffinityPartitioning = [] {
  const char *Flag =
      std::getenv("SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING");
  return Flag ? std::atoi(Flag) != 0 : false;
}();

template <typename RangeTy, typename ElemTy>
bool contains(RangeTy &&Range, const ElemTy &Elem) {
  return std::find(Range.begin(), Range.end(), Elem) != Range.end();
}

bool isPartitionableBy(device &Dev, info::partition_property Prop) {
  return contains(Dev.get_info<info::device::partition_properties>(), Prop);
}

bool isPartitionableByCSlice(device &Dev) {
  return isPartitionableBy(
      Dev, info::partition_property::ext_intel_partition_by_cslice);
}

bool isPartitionableByAffinityDomain(device &Dev) {
  return isPartitionableBy(
      Dev, info::partition_property::partition_by_affinity_domain);
}

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

    assert(isPartitionableByAffinityDomain(d));
    assert(!isPartitionableByCSlice(d));
    {
      try {
        std::ignore = d.create_sub_devices<
            info::partition_property::ext_intel_partition_by_cslice>();
        assert(false && "Expected an exception to be thrown earlier!");
      } catch (sycl::exception &e) {
        assert(e.code() == errc::feature_not_supported);
      }
    }

    auto sub_devices = d.create_sub_devices<
        info::partition_property::partition_by_affinity_domain>(
        info::partition_affinity_domain::next_partitionable);
    device &sub_device = sub_devices[1];
    assert(isPartitionableByAffinityDomain(sub_device) ==
           ExposeCSliceInAffinityPartitioning);
    assert(isPartitionableByCSlice(sub_device));
    assert(sub_device.get_info<info::device::partition_type_property>() ==
           info::partition_property::partition_by_affinity_domain);

    {
      try {
        std::ignore = sub_device.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
        assert(ExposeCSliceInAffinityPartitioning &&
               "Expected an exception to be thrown earlier!");
      } catch (sycl::exception &e) {
        assert(e.code() == errc::feature_not_supported);
      }
    }

    auto VerifySubSubDevice = [&](auto &sub_sub_devices) {
      device &sub_sub_device = sub_sub_devices[1];
      assert(sub_sub_devices.size() == NumCSlices);
      assert(!isPartitionableByAffinityDomain(sub_sub_device));
      assert(!isPartitionableByCSlice(sub_sub_device));

      // Note that we still report this sub-sub-device as created via
      // partitioning by cslice even if it was partition by affinity domain.
      // This is a known limitation that we won't address as the whole code path
      // (exposing CSlice as sub-devices via partitioning by affinity domaing
      // using SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING
      // environment variable) is deprecated  and is going to be removed.
      assert(sub_sub_device.get_info<info::device::partition_type_property>() ==
             info::partition_property::ext_intel_partition_by_cslice);

      assert(sub_sub_device.get_info<info::device::max_compute_units>() *
                 NumCSlices ==
             sub_device.get_info<info::device::max_compute_units>());

      {
        queue q{sub_device};
        q.single_task([=]() {});
      }
      {
        queue q{sub_sub_device};
        q.single_task([=]() {});
      }
      // CHECK-PVC:          [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
      // CHECK-PVC:          [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
      // CHECK-PVC-AFFINITY: [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
      // CHECK-PVC-AFFINITY: [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
    };
    {
      auto sub_sub_devices = sub_device.create_sub_devices<
          info::partition_property::ext_intel_partition_by_cslice>();
      VerifySubSubDevice(sub_sub_devices);
    }

    if (ExposeCSliceInAffinityPartitioning) {
      auto sub_sub_devices = sub_device.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
      VerifySubSubDevice(sub_sub_devices);
    }
  } else {
    // Make FileCheck pass.
    std::cout << "Fake ZE_DEBUG output for FileCheck:" << std::endl;
    // clang-format off
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])" << std::endl;
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])" << std::endl;
    if (ExposeCSliceInAffinityPartitioning) {
      std::cout << "[getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])" << std::endl;
      std::cout << "[getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])" << std::endl;
    }
    // clang-format on
  }
  std::cout << "Test PVC End" << std::endl;
  // CHECK-PVC: Test PVC End
}

void test_non_pvc(device d) {
  bool IsPVC = [&]() {
    if (!d.has(aspect::ext_intel_device_id))
      return false;
    return (d.get_info<ext::intel::info::device::device_id>() & 0xff0) == 0xbd0;
  }();

  if (IsPVC)
    return;

  // Non-PVC devices are not partitionable by CSlice at any level of
  // partitioning.

  while (true) {
    assert(!isPartitionableByCSlice(d));

    if (!isPartitionableByAffinityDomain(d))
      // No more sub-devices.
      break;

    auto sub_devices = d.create_sub_devices<
        info::partition_property::partition_by_affinity_domain>(
        info::partition_affinity_domain::next_partitionable);
    d = sub_devices[0];
  }
}

int main() {
  device d;

  test_pvc(d);
  test_non_pvc(d);

  return 0;
}
