// This test checks to see if every aspect and sub-group size declared in the
// device config file is supported by the device. Note this does not mean
// check that the device config file is exhaustive, only that the device
// supports everything it declares. However, this test does print out any
// aspects that are supported by the device but not declared in the device
// config file.

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: Accelerator is not supported by sycl_ext_oneapi_device_architecture
// REQUIRES: device-config-file
// RUN: %{build} -o %t.out %device_config_file_include_flag
// RUN: %{run} %t.out
#include <map>

#include <sycl/detail/core.hpp>
#include <llvm/SYCLLowerIR/DeviceConfigFile.hpp>

#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)

using namespace sycl;

const char *getArchName(const device &Device) {
  namespace syclex = sycl::ext::oneapi::experimental;
  auto arch = Device.get_info<syclex::info::device::architecture>();
  switch (arch) {
#define __SYCL_ARCHITECTURE(ARCH, VAL)                                         \
  case syclex::architecture::ARCH:                                             \
    return #ARCH;
#define __SYCL_ARCHITECTURE_ALIAS(ARCH, VAL)
#include <sycl/ext/oneapi/experimental/device_architecture.def>
#undef __SYCL_ARCHITECTURE
#undef __SYCL_ARCHITECTURE_ALIAS
  }
  return "unknown";
}

// checks if a container contains a specific element
template <typename Container, typename T>
bool contains(const Container &c, const T &elem) {
  return std::find(c.begin(), c.end(), elem) != c.end();
}

std::string_view getAspectName(aspect asp) {
  switch (asp) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case aspect::ASPECT:                                                         \
    return #ASPECT;
#include <sycl/info/aspects.def>
#undef __SYCL_ASPECT
  }
  return "unknown";
}

aspect getAspectByName(std::string_view name) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  if (name == #ASPECT)                                                         \
    return aspect::ASPECT;
#include <sycl/info/aspects.def>
  throw std::invalid_argument("Unknown aspect name");
}

int main() {
  // Get the device arch
  queue q;
  auto dev = q.get_device();
  auto device_name = getArchName(dev);

  auto TargetInfo = DeviceConfigFile::TargetTable.find(device_name);
  if (TargetInfo == DeviceConfigFile::TargetTable.end()) {
    std::cout << "No aspects found for device " << device_name << std::endl;
    return 1;
  }

  // Check aspects consistency
  int nAspectInconsistencies = 0;
  std::cout << "Checking consistency of aspects for device " << device_name
            << "...\n";

  auto supportedAspects = dev.get_info<info::device::aspects>();
  auto deviceConfigAspectNames = TargetInfo->second.aspects;
  std::vector<aspect> deviceConfigAspects;
  for (auto aspectName : deviceConfigAspectNames) {
    deviceConfigAspects.push_back(getAspectByName(aspectName));
  }

  for (auto asp : deviceConfigAspects) {
    if (!contains(supportedAspects, asp)) {
      std::cout << "error: " << device_name << " does not support aspect "
                << getAspectName(asp)
                << " but it is declared in the device config file\n";
      ++nAspectInconsistencies;
    }
  }
  for (auto asp : supportedAspects) {
    if (!contains(deviceConfigAspects, asp)) {
      std::cout << "note: the device " << device_name << " supports aspect "
                << getAspectName(asp)
                << " but it is not declared in the device config file\n";
      // Not necessarily an error, so we won't increment n_fail
    }
  }

  if (nAspectInconsistencies == 0)
    std::cout << "All aspects are consistent\n";

  // Check sub-group sizes consistency
  int nSubGroupSizeInconsistencies = 0;
  std::cout << "Checking consistency of sub-group sizes for device "
            << device_name << "...\n";

  auto supportedSubGroupSizes = dev.get_info<info::device::sub_group_sizes>();
  auto deviceConfigSubGroupSizes = TargetInfo->second.subGroupSizes;

  for (auto size : deviceConfigSubGroupSizes) {
    if (!contains(supportedSubGroupSizes, size)) {
      std::cout << "error: " << device_name
                << " does not support sub-group size " << size
                << " but it is declared in the device config file\n";
      ++nSubGroupSizeInconsistencies;
    }
  }
  for (auto size : supportedSubGroupSizes) {
    if (!contains(deviceConfigSubGroupSizes, size)) {
      std::cout << "note: the device " << device_name
                << " supports sub-group size " << size
                << " but it is not declared in the device config file\n";
      // Not necessarily an error, so we won't increment n_fail
    }
  }

  if (nSubGroupSizeInconsistencies == 0)
    std::cout << "All sub-group sizes are consistent\n";

  return nAspectInconsistencies + nSubGroupSizeInconsistencies;
}

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
