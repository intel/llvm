// This test checks to see if every aspect and sub-group size declared in the
// device config file is supported by the device. Note this does not mean
// check that the device config file is exhaustive, only that the device
// supports everything it declares. However, this test does print out any
// aspects that are supported by the device but not declared in the device
// config file.

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: Accelerator is not supported by
// sycl_ext_oneapi_device_architecture.
// REQUIRES: device-config-file
// RUN: %{build} -o %t.out %device_config_file_include_flag
// RUN: %{run} %t.out
#include <map>

#include <llvm/SYCLLowerIR/DeviceConfigFile.hpp>
#include <sycl/detail/core.hpp>

#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)

using namespace sycl;

const char *getArchName(const device &Device) {
  namespace syclex = sycl::ext::oneapi::experimental;
  auto Arch = Device.get_info<syclex::info::device::architecture>();
  switch (Arch) {
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

// Checks if a container contains a specific element
template <typename Container, typename T>
bool contains(const Container &C, const T &Elem) {
  return std::find(C.begin(), C.end(), Elem) != C.end();
}

std::string_view getAspectName(aspect Asp) {
  switch (Asp) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case aspect::ASPECT:                                                         \
    return #ASPECT;
#include <sycl/info/aspects.def>
#undef __SYCL_ASPECT
  }
  return "unknown";
}

aspect getAspectByName(std::string_view Name) {
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  if (Name == #ASPECT)                                                         \
    return aspect::ASPECT;
#include <sycl/info/aspects.def>
  throw std::invalid_argument("Unknown aspect name");
}

int main() {
  // Get the device arch
  queue Q;
  auto Dev = Q.get_device();
  auto DeviceName = getArchName(Dev);

  auto TargetInfo = DeviceConfigFile::TargetTable.find(DeviceName);
  if (TargetInfo == DeviceConfigFile::TargetTable.end()) {
    std::cout << "No aspects found for device " << DeviceName << "\n";
    return 1;
  }

  // Check aspects consistency
  int NAspectInconsistencies = 0;
  std::cout << "Checking consistency of aspects for device " << DeviceName
            << "...\n";

  auto SupportedAspects = Dev.get_info<info::device::aspects>();
  auto DeviceConfigAspectNames = TargetInfo->second.aspects;
  std::vector<aspect> DeviceConfigAspects;
  for (auto AspectName : DeviceConfigAspectNames) {
    DeviceConfigAspects.push_back(getAspectByName(AspectName));
  }

  for (auto Asp : DeviceConfigAspects) {
    if (!contains(SupportedAspects, Asp)) {
      std::cout << "error: " << DeviceName << " does not support aspect "
                << getAspectName(Asp)
                << " but it is declared in the device config file\n";
      ++NAspectInconsistencies;
    }
  }
  for (auto Asp : SupportedAspects) {
    if (!contains(DeviceConfigAspects, Asp)) {
      std::cout << "note: the device " << DeviceName << " supports aspect "
                << getAspectName(Asp)
                << " but it is not declared in the device config file\n";
      // Not necessarily an error, so we won't increment n_fail
    }
  }

  if (NAspectInconsistencies == 0)
    std::cout << "All aspects are consistent\n";

  // Check sub-group sizes consistency
  int NSubGroupSizeInconsistencies = 0;
  std::cout << "Checking consistency of sub-group sizes for device "
            << DeviceName << "...\n";

  auto SupportedSubGroupSizes = Dev.get_info<info::device::sub_group_sizes>();
  auto DeviceConfigSubGroupSizes = TargetInfo->second.subGroupSizes;

  for (auto Size : DeviceConfigSubGroupSizes) {
    if (!contains(SupportedSubGroupSizes, Size)) {
      std::cout << "error: " << DeviceName
                << " does not support sub-group size " << Size
                << " but it is declared in the device config file\n";
      ++NSubGroupSizeInconsistencies;
    }
  }
  for (auto Size : SupportedSubGroupSizes) {
    if (!contains(DeviceConfigSubGroupSizes, Size)) {
      std::cout << "note: the device " << DeviceName
                << " supports sub-group size " << Size
                << " but it is not declared in the device config file\n";
      // Not necessarily an error, so we won't increment n_fail
    }
  }

  if (NSubGroupSizeInconsistencies == 0)
    std::cout << "All sub-group sizes are consistent\n";

  return NAspectInconsistencies + NSubGroupSizeInconsistencies;
}

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
