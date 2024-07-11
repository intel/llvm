//==------- kernel_bundle.cpp - SYCL kernel_bundle and free functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_binary_image.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_compiler/kernel_compiler_opencl.hpp>
#include <detail/kernel_compiler/kernel_compiler_sycl.hpp>
#include <detail/kernel_id_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <cstddef>
#include <set>
#include <vector>

namespace sycl {
inline namespace _V1 {

kernel_id::kernel_id(const char *Name)
    : impl(std::make_shared<detail::kernel_id_impl>(Name)) {}

const char *kernel_id::get_name() const noexcept { return impl->get_name(); }

namespace detail {

////////////////////////////
/////  device_image_plain
///////////////////////////

bool device_image_plain::has_kernel(const kernel_id &KernelID) const noexcept {
  return impl->has_kernel(KernelID);
}

bool device_image_plain::has_kernel(const kernel_id &KernelID,
                                    const device &Dev) const noexcept {
  return impl->has_kernel(KernelID, Dev);
}

pi_native_handle device_image_plain::getNative() const {
  return impl->getNative();
}

////////////////////////////
///// kernel_bundle_plain
///////////////////////////

bool kernel_bundle_plain::empty() const noexcept { return impl->empty(); }

backend kernel_bundle_plain::get_backend() const noexcept {
  return impl->get_backend();
}

context kernel_bundle_plain::get_context() const noexcept {
  return impl->get_context();
}

std::vector<device> kernel_bundle_plain::get_devices() const noexcept {
  return impl->get_devices();
}

std::vector<kernel_id> kernel_bundle_plain::get_kernel_ids() const {
  return impl->get_kernel_ids();
}

bool kernel_bundle_plain::contains_specialization_constants() const noexcept {
  return impl->contains_specialization_constants();
}

bool kernel_bundle_plain::native_specialization_constant() const noexcept {
  return impl->native_specialization_constant();
}

kernel kernel_bundle_plain::get_kernel(const kernel_id &KernelID) const {
  return impl->get_kernel(KernelID, impl);
}

const device_image_plain *kernel_bundle_plain::begin() const {
  return impl->begin();
}

const device_image_plain *kernel_bundle_plain::end() const {
  return impl->end();
}

bool kernel_bundle_plain::has_kernel(const kernel_id &KernelID) const noexcept {
  return impl->has_kernel(KernelID);
}

bool kernel_bundle_plain::has_kernel(const kernel_id &KernelID,
                                     const device &Dev) const noexcept {
  return impl->has_kernel(KernelID, Dev);
}

bool kernel_bundle_plain::has_specialization_constant_impl(
    const char *SpecName) const noexcept {
  return impl->has_specialization_constant(SpecName);
}

void kernel_bundle_plain::set_specialization_constant_impl(
    const char *SpecName, void *Value, size_t Size) noexcept {
  impl->set_specialization_constant_raw_value(SpecName, Value, Size);
}

void kernel_bundle_plain::get_specialization_constant_impl(
    const char *SpecName, void *Value) const noexcept {
  impl->get_specialization_constant_raw_value(SpecName, Value);
}

bool kernel_bundle_plain::is_specialization_constant_set(
    const char *SpecName) const noexcept {
  return impl->is_specialization_constant_set(SpecName);
}

bool kernel_bundle_plain::ext_oneapi_has_kernel(detail::string_view name) {
  return impl->ext_oneapi_has_kernel(name.data());
}

kernel kernel_bundle_plain::ext_oneapi_get_kernel(detail::string_view name) {
  return impl->ext_oneapi_get_kernel(name.data(), impl);
}

//////////////////////////////////
///// sycl::detail free functions
//////////////////////////////////

const std::vector<device>
removeDuplicateDevices(const std::vector<device> &Devs) {
  std::vector<device> UniqueDevices;

  // Building a new vector with unique elements and keep original order
  std::unordered_set<device> UniqueDeviceSet;
  for (const device &Dev : Devs)
    if (UniqueDeviceSet.insert(Dev).second)
      UniqueDevices.push_back(Dev);

  return UniqueDevices;
}

kernel_id get_kernel_id_impl(string_view KernelName) {
  return detail::ProgramManager::getInstance().getSYCLKernelID(
      KernelName.data());
}

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, State);
}

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       const std::vector<kernel_id> &KernelIDs,
                       bundle_state State) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, KernelIDs,
                                                      State);
}

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State, const DevImgSelectorImpl &Selector) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, Selector,
                                                      State);
}

detail::KernelBundleImplPtr
get_empty_interop_kernel_bundle_impl(const context &Ctx,
                                     const std::vector<device> &Devs) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs);
}

std::shared_ptr<detail::kernel_bundle_impl>
join_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles,
          bundle_state State) {
  return std::make_shared<detail::kernel_bundle_impl>(Bundles, State);
}

bool has_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                            bundle_state State) {
  // Check that all requested devices are associated with the context
  const bool AllDevicesInTheContext = checkAllDevicesAreInContext(Devs, Ctx);
  if (Devs.empty() || !AllDevicesInTheContext)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Not all devices are associated with the context or "
                          "vector of devices is empty");

  if (bundle_state::input == State &&
      !checkAllDevicesHaveAspect(Devs, aspect::online_compiler))
    return false;
  if (bundle_state::object == State &&
      !checkAllDevicesHaveAspect(Devs, aspect::online_linker))
    return false;

  const std::vector<device_image_plain> DeviceImages =
      detail::ProgramManager::getInstance()
          .getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, State);

  return (bool)DeviceImages.size();
}

bool has_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                            const std::vector<kernel_id> &KernelIds,
                            bundle_state State) {
  // Check that all requested devices are associated with the context
  const bool AllDevicesInTheContext = checkAllDevicesAreInContext(Devs, Ctx);

  if (Devs.empty() || !AllDevicesInTheContext)
    throw sycl::exception(make_error_code(errc::invalid),
                          "Not all devices are associated with the context or "
                          "vector of devices is empty");

  bool DeviceHasRequireAspectForState = true;
  if (bundle_state::input == State) {
    DeviceHasRequireAspectForState =
        std::all_of(Devs.begin(), Devs.end(), [](const device &Dev) {
          return Dev.has(aspect::online_compiler);
        });
  } else if (bundle_state::object == State) {
    DeviceHasRequireAspectForState =
        std::all_of(Devs.begin(), Devs.end(), [](const device &Dev) {
          return Dev.has(aspect::online_linker);
        });
  }

  if (!DeviceHasRequireAspectForState)
    return false;

  const std::vector<device_image_plain> DeviceImages =
      detail::ProgramManager::getInstance()
          .getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, State);

  std::set<kernel_id, LessByNameComp> CombinedKernelIDs;
  for (const device_image_plain &DeviceImage : DeviceImages) {
    const std::shared_ptr<device_image_impl> &DeviceImageImpl =
        getSyclObjImpl(DeviceImage);

    CombinedKernelIDs.insert(DeviceImageImpl->get_kernel_ids_ptr()->begin(),
                             DeviceImageImpl->get_kernel_ids_ptr()->end());
  }

  const bool AllKernelIDsRepresented =
      std::all_of(KernelIds.begin(), KernelIds.end(),
                  [&CombinedKernelIDs](const kernel_id &KernelID) {
                    return CombinedKernelIDs.count(KernelID);
                  });

  return AllKernelIDsRepresented;
}

std::shared_ptr<detail::kernel_bundle_impl>
compile_impl(const kernel_bundle<bundle_state::input> &InputBundle,
             const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(
      InputBundle, Devs, PropList, bundle_state::object);
}

std::shared_ptr<detail::kernel_bundle_impl>
link_impl(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
          const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(ObjectBundles, Devs,
                                                      PropList);
}

std::shared_ptr<detail::kernel_bundle_impl>
build_impl(const kernel_bundle<bundle_state::input> &InputBundle,
           const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(
      InputBundle, Devs, PropList, bundle_state::executable);
}

// This function finds intersection of associated devices in common for all
// bundles
std::vector<sycl::device> find_device_intersection(
    const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles) {
  std::vector<sycl::device> IntersectDevices;
  std::vector<unsigned int> DevsCounters;
  std::map<device, unsigned int, LessByHash<device>> DevCounters;
  for (const sycl::kernel_bundle<bundle_state::object> &ObjectBundle :
       ObjectBundles)
    // Increment counter in "DevCounters" each time a device is seen
    for (const sycl::device &Device : ObjectBundle.get_devices())
      DevCounters[Device]++;

  // If some device counter is less than ObjectBundles.size() then some bundle
  // doesn't have it - do not add such a device to the final result
  for (const std::pair<const device, unsigned int> &It : DevCounters)
    if (ObjectBundles.size() == It.second)
      IntersectDevices.push_back(It.first);

  return IntersectDevices;
}

} // namespace detail

//////////////////////////
///// sycl free functions
//////////////////////////

std::vector<kernel_id> get_kernel_ids() {
  return detail::ProgramManager::getInstance().getAllSYCLKernelIDs();
}

bool is_compatible(const std::vector<kernel_id> &KernelIDs, const device &Dev) {
  if (KernelIDs.empty())
    return true;
  // TODO: also need to check that the architecture specified by the
  // "-fsycl-targets" flag matches the device when we are able to get the
  // device's arch.
  auto doesImageTargetMatchDevice = [](const device &Dev,
                                       const detail::RTDeviceBinaryImage &Img) {
    const char *Target = Img.getRawData().DeviceTargetSpec;
    auto BE = Dev.get_backend();
    if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0) {
      return (BE == sycl::backend::opencl ||
              BE == sycl::backend::ext_oneapi_level_zero);
    } else if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) ==
               0) {
      return Dev.is_cpu();
    } else if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) ==
               0) {
      return Dev.is_gpu() && (BE == sycl::backend::opencl ||
                              BE == sycl::backend::ext_oneapi_level_zero);
    } else if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) ==
               0) {
      return Dev.is_accelerator();
    } else if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_NVPTX64) == 0) {
      return BE == sycl::backend::ext_oneapi_cuda;
    } else if (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_AMDGCN) == 0) {
      return BE == sycl::backend::ext_oneapi_hip;
    }

    return false;
  };

  // One kernel may be contained in several binary images depending on the
  // number of targets. This kernel is compatible with the device if there is
  // at least one image (containing this kernel) whose aspects are supported by
  // the device and whose target matches the device.
  for (const auto &KernelID : KernelIDs) {
    std::set<detail::RTDeviceBinaryImage *> BinImages =
        detail::ProgramManager::getInstance().getRawDeviceImages({KernelID});

    if (std::none_of(BinImages.begin(), BinImages.end(),
                     [&](const detail::RTDeviceBinaryImage *Img) {
                       return doesDevSupportDeviceRequirements(Dev, *Img) &&
                              doesImageTargetMatchDevice(Dev, *Img);
                     }))
      return false;
  }

  return true;
}

/////////////////////////
// * kernel_compiler extension *
/////////////////////////
namespace ext::oneapi::experimental {

using source_kb = kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using exe_kb = kernel_bundle<bundle_state::executable>;
using kernel_bundle_impl = sycl::detail::kernel_bundle_impl;

namespace detail {

/////////////////////////
// syclex::detail::is_source_kernel_bundle_supported
/////////////////////////
bool is_source_kernel_bundle_supported(backend BE, source_language Language) {
  // Support is limited to the opencl and level_zero backends.
  bool BE_Acceptable = (BE == sycl::backend::ext_oneapi_level_zero) ||
                       (BE == sycl::backend::opencl);
  if (BE_Acceptable) {
    if (Language == source_language::opencl) {
      return detail::OpenCLC_Compilation_Available();
    } else if (Language == source_language::spirv) {
      return true;
    } else if (Language == source_language::sycl) {
      return detail::SYCL_Compilation_Available();
    }
  }

  // otherwise
  return false;
}

/////////////////////////
// syclex::detail::create_kernel_bundle_from_source
/////////////////////////

using include_pairs_t = std::vector<std::pair<std::string, std::string>>;
using include_pairs_view_t = std::vector<
    std::pair<sycl::detail::string_view, sycl::detail::string_view>>;

source_kb
make_kernel_bundle_from_source(const context &SyclContext,
                               source_language Language,
                               sycl::detail::string_view SourceView,
                               include_pairs_view_t IncludePairViews) {
  // TODO: if we later support a "reason" why support isn't present
  // (like a missing shared library etc.) it'd be nice to include it in
  // the exception message here.
  std::string Source{SourceView.data()};
  include_pairs_t IncludePairs;
  size_t n = IncludePairViews.size();
  IncludePairs.reserve(n);
  for (auto &p : IncludePairViews)
    IncludePairs.push_back({p.first.data(), p.second.data()});

  backend BE = SyclContext.get_backend();
  if (!is_source_kernel_bundle_supported(BE, Language))
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel_bundle creation from source not supported");

  // throw if include not supported?   awaiting guidance
  // if(!IncludePairs.empty() && is_include_supported(Languuage)){ throw invalid
  // }

  std::shared_ptr<kernel_bundle_impl> KBImpl =
      std::make_shared<kernel_bundle_impl>(SyclContext, Language, Source,
                                           IncludePairs);
  return sycl::detail::createSyclObjFromImpl<source_kb>(KBImpl);
}

source_kb make_kernel_bundle_from_source(const context &SyclContext,
                                         source_language Language,
                                         const std::vector<std::byte> &Bytes,
                                         include_pairs_view_t IncludePairs) {
  (void)IncludePairs;
  backend BE = SyclContext.get_backend();
  if (!is_source_kernel_bundle_supported(BE, Language))
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel_bundle creation from source not supported");

  std::shared_ptr<kernel_bundle_impl> KBImpl =
      std::make_shared<kernel_bundle_impl>(SyclContext, Language, Bytes);
  return sycl::detail::createSyclObjFromImpl<source_kb>(KBImpl);
}

/////////////////////////
// syclex::detail::build_from_source(source_kb) => exe_kb
/////////////////////////

exe_kb build_from_source(
    source_kb &SourceKB, const std::vector<device> &Devices,
    const std::vector<sycl::detail::string_view> &BuildOptions,
    sycl::detail::string *LogView,
    const std::vector<sycl::detail::string_view> &RegisteredKernelNames) {
  std::vector<std::string> Options;
  for (const sycl::detail::string_view option : BuildOptions)
    Options.push_back(option.data());

  std::vector<std::string> KernelNames;
  for (const sycl::detail::string_view name : RegisteredKernelNames)
    KernelNames.push_back(name.data());

  std::string Log;
  std::string *LogPtr = nullptr;
  if (LogView)
    LogPtr = &Log;
  std::vector<device> UniqueDevices =
      sycl::detail::removeDuplicateDevices(Devices);
  std::shared_ptr<kernel_bundle_impl> sourceImpl = getSyclObjImpl(SourceKB);
  std::shared_ptr<kernel_bundle_impl> KBImpl = sourceImpl->build_from_source(
      UniqueDevices, Options, LogPtr, KernelNames);
  auto result = sycl::detail::createSyclObjFromImpl<exe_kb>(KBImpl);
  if (LogView)
    *LogView = Log;
  return result;
}

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
