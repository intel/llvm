// REQUIRES: level_zero
// REQUIRES: aspect-ext_intel_device_id

// RUN: %{build} -o %t.out

// TODO - at this time ZEX_NUMBER_OF_CCS is not working with FLAT hierachy,
// which is the new default on PVC.  Once it is supported, we'll test on both.
// In the interim, these are the environment vars that must be set to get cslice
// or the extra level of partition_by_affinity_domain with the "EXPOSE_" env
// var.
// DEFINE: %{setup_env} = env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 ZEX_NUMBER_OF_CCS=0:4

// RUN: %{setup_env} env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-PVC

// RUN: %{setup_env} env SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING=1 \
// RUN:  UR_L0_DEBUG=1 %{run} %t.out  2>&1 | FileCheck %s --check-prefixes=CHECK-PVC

// Same, but without using immediate commandlists:

// RUN: %{setup_env} env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0  \
// RUN:   UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-PVC

// RUN: %{setup_env} env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING=1 \
// RUN:  UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-PVC

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

bool IsPVC(device &d) {
  uint32_t masked_device_id =
      d.get_info<ext::intel::info::device::device_id>() & 0xff0;
  return masked_device_id == 0xbd0 || masked_device_id == 0xb60;
}

void test_pvc(device &d) {
  std::cout << "Test PVC Begin" << std::endl;
  // CHECK-PVC: Test PVC Begin
  std::cout << "IsPVC: " << IsPVC(d) << std::endl;
  if (IsPVC(d)) {
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
    };

    // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
    // CHECK-PVC: [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
    if (ExposeCSliceInAffinityPartitioning) {
      auto sub_sub_devices = sub_device.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
      VerifySubSubDevice(sub_sub_devices);
    } else {
      auto sub_sub_devices = sub_device.create_sub_devices<
          info::partition_property::ext_intel_partition_by_cslice>();
      VerifySubSubDevice(sub_sub_devices);
    }
  } else {
    // Make FileCheck pass.
    std::cout << "Fake UR_L0_DEBUG output for FileCheck:" << std::endl;
    // clang-format off
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])" << std::endl;
    std::cout << "[getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])" << std::endl;
    // clang-format on
  }
  std::cout << "Test PVC End" << std::endl;
  // CHECK-PVC: Test PVC End
}

void test_non_pvc(device &d) {
  if (IsPVC(d))
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
