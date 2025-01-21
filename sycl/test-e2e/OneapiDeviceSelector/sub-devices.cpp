// works on PVC or ATS, however, the gpu-intel-gen12 identifier seems to
// not be working with  REQUIRES ,
// TODO : fix gpu-intel-XXXX identifiers for REQUIRE and UNSUPPORTED

// REQUIRES: gpu

// open question on whether this will work with CUDA or Hip.

// RUN: %{build} -o %t.out

// select sub-devices
//   ONEAPI_DEVICE_SELECTOR=(any backend):(any
//   device).(all-the-sub-devices).(all-sub-sub-devices)
// RUN: env ONEAPI_DEVICE_SELECTOR="*:*.*" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:*.*.*" %{run-unfiltered-devices} %t.out

// select root devices and pass arg to test so it knows.
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out 1
// RUN: %{run-unfiltered-devices} %t.out 1

#include <sycl/detail/core.hpp>
using namespace sycl;

int main(int Argc, const char *Argv[]) {
  std::vector<device> devices = device::get_devices();
  std::cout << devices.size() << " devices found." << std::endl;
  // root devices, no matter the selector should not have parents.
  for (const device &dev : devices) {
    std::cout << "name: " << dev.get_info<info::device::name>() << std::endl;
    try {
      // unlike sub-devices gotten from device.get_info<device::parent_device>
      // sub-devices gotten via ONEAPI_DEVICE_SELECTOR pretend to be root
      // devices and do NOT have parents.
      device parentDev = dev.get_info<info::device::parent_device>();
      std::cout << "unexpected parent name: "
                << parentDev.get_info<info::device::name>() << std::endl;
      assert(false && "we should not be here. asking for a parent should throw "
                      "an exception");

    } catch (sycl::exception &e) {
      std::cout << "Yay. No parent device gotten" << std::endl;
    }
  }
  // any argument indicates "normal" usage, no sub-devices promoted by
  // ONEAPI_DEVICE_SELECTOR
  if (Argc > 1) {
    constexpr info::partition_property partitionProperty =
        info::partition_property::partition_by_affinity_domain;
    constexpr info::partition_affinity_domain affinityDomain =
        info::partition_affinity_domain::next_partitionable;
    device d = device(gpu_selector_v);
    try {
      std::vector<device> sub_devices =
          d.create_sub_devices<partitionProperty>(affinityDomain);
      for (const device &sub_dev : sub_devices) {
        std::cout << "child name: " << sub_dev.get_info<info::device::name>()
                  << std::endl;
        try {
          // sub-devices gotten from device.get_info<device::parent_device>
          // should have parents.
          device parentDev = sub_dev.get_info<info::device::parent_device>();
          std::cout << "cool parent name: "
                    << parentDev.get_info<info::device::name>() << std::endl;
        } catch (sycl::exception &e) {
          std::cout << "exception: " << e.what() << std::endl;
          assert(false && "we should not be here. asking for a parent of a "
                          "sub-device should NOT throw an exception ");
        }
      }
    } catch (sycl::exception &e) {
      // whatever GPU device we are on does not support sub-devices at all.
      // Nothing to do.
      std::cout << "this device " << d.get_info<info::device::name>()
                << " does not support sub-devices. Nothing tested."
                << std::endl;
    }
  }
  return 0;
}
