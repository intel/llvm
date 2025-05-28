//==------- kernel_bundle.cpp - SYCL kernel_bundle and free functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/compiler.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_compiler/kernel_compiler_opencl.hpp>
#include <detail/kernel_compiler/kernel_compiler_sycl.hpp>
#include <detail/kernel_id_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <cstddef>
#include <set>
#include <string_view>
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

ur_native_handle_t device_image_plain::getNative() const {
  return impl->getNative();
}

backend device_image_plain::ext_oneapi_get_backend_impl() const noexcept {
  return impl->get_context().get_backend();
}

std::pair<const std::byte *, const std::byte *>
device_image_plain::ext_oneapi_get_backend_content_view_impl() const {
  return std::make_pair(
      reinterpret_cast<const std::byte *>(
          impl->get_bin_image_ref()->getRawData().BinaryStart),
      reinterpret_cast<const std::byte *>(
          impl->get_bin_image_ref()->getRawData().BinaryEnd));
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
  return impl->ext_oneapi_has_kernel(std::string(std::string_view(name)));
}

kernel kernel_bundle_plain::ext_oneapi_get_kernel(detail::string_view name) {
  return impl->ext_oneapi_get_kernel(std::string(std::string_view(name)),
      impl);
}

detail::string
kernel_bundle_plain::ext_oneapi_get_raw_kernel_name(detail::string_view name) {
  return detail::string{impl->ext_oneapi_get_raw_kernel_name(
      std::string(std::string_view(name)))};
}

bool kernel_bundle_plain::ext_oneapi_has_device_global(
    detail::string_view name) {
  return impl->ext_oneapi_has_device_global(
      std::string(std::string_view(name)));
}

void *kernel_bundle_plain::ext_oneapi_get_device_global_address(
    detail::string_view name, const device &dev) {
  return impl->ext_oneapi_get_device_global_address(
      std::string(std::string_view(name)), dev);
}

size_t kernel_bundle_plain::ext_oneapi_get_device_global_size(
    detail::string_view name) {
  return impl->ext_oneapi_get_device_global_size(std::string(std::string_view(name)));
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
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    std::string(
#endif
      std::string_view(KernelName)
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
    )
#endif
    );
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

  const std::vector<DevImgPlainWithDeps> DeviceImages =
      detail::ProgramManager::getInstance()
          .getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, State);

  return !DeviceImages.empty();
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

  const std::vector<DevImgPlainWithDeps> DeviceImagesWithDeps =
      detail::ProgramManager::getInstance()
          .getSYCLDeviceImagesWithCompatibleState(Ctx, Devs, State);

  std::set<kernel_id, LessByNameComp> CombinedKernelIDs;
  for (const DevImgPlainWithDeps &DeviceImageWithDeps : DeviceImagesWithDeps) {
    for (const device_image_plain &DeviceImage : DeviceImageWithDeps) {
      const std::shared_ptr<device_image_impl> &DeviceImageImpl =
          getSyclObjImpl(DeviceImage);

      CombinedKernelIDs.insert(DeviceImageImpl->get_kernel_ids_ptr()->begin(),
                               DeviceImageImpl->get_kernel_ids_ptr()->end());
    }
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
  std::vector<kernel_id> ids =
      detail::ProgramManager::getInstance().getAllSYCLKernelIDs();
  // Filter out kernel ids coming from RTC kernels in order to be
  // spec-compliant. Kernel ids from RTC are prefixed with rtc_NUM$, so looking
  // for '$' should be enough.
  ids.erase(std::remove_if(ids.begin(), ids.end(),
                           [](kernel_id id) {
                             std::string_view sv(id.get_name());
                             return sv.find('$') != std::string_view::npos;
                           }),
            ids.end());
  return ids;
}

bool is_compatible(const std::vector<kernel_id> &KernelIDs, const device &Dev) {
  if (KernelIDs.empty())
    return true;
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
                              doesImageTargetMatchDevice(
                                  *Img, getSyclObjImpl(Dev).get());
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
using obj_kb = kernel_bundle<bundle_state::object>;
using exe_kb = kernel_bundle<bundle_state::executable>;
using kernel_bundle_impl = sycl::detail::kernel_bundle_impl;

namespace detail {

/////////////////////////
// syclex::detail::is_source_kernel_bundle_supported
/////////////////////////

bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const std::vector<device_impl *> &DeviceImplVec) {
  backend BE = DeviceImplVec[0]->getBackend();
  // Support is limited to the opencl and level_zero backends.
  bool BE_Acceptable = (BE == sycl::backend::ext_oneapi_level_zero) ||
                       (BE == sycl::backend::opencl);
  if (!BE_Acceptable)
    return false;

  if (Language == source_language::spirv) {
    return true;
  } else if (Language == source_language::sycl) {
    return detail::SYCL_JIT_Compilation_Available();
  } else if (Language == source_language::opencl) {
    if (DeviceImplVec.empty())
      return false;

    const AdapterPtr &Adapter = DeviceImplVec[0]->getAdapter();
    std::vector<uint32_t> IPVersionVec;
    IPVersionVec.reserve(DeviceImplVec.size());

    std::transform(DeviceImplVec.begin(), DeviceImplVec.end(),
                   std::back_inserter(IPVersionVec), [&](device_impl *Dev) {
                     uint32_t ipVersion = 0;
                     ur_device_handle_t DeviceHandle = Dev->getHandleRef();
                     Adapter->call<UrApiKind::urDeviceGetInfo>(
                         DeviceHandle, UR_DEVICE_INFO_IP_VERSION,
                         sizeof(uint32_t), &ipVersion, nullptr);
                     return ipVersion;
                   });

    return detail::OpenCLC_Compilation_Available(IPVersionVec);
  }

  // otherwise
  return false;
}

bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const context &Ctx) {
  const std::vector<sycl::device> Devices = Ctx.get_devices();
  std::vector<device_impl *> DeviceImplVec;
  DeviceImplVec.reserve(Devices.size());
  std::transform(Devices.begin(), Devices.end(),
                 std::back_inserter(DeviceImplVec),
                 [](const sycl::device &dev) {
                   return &*sycl::detail::getSyclObjImpl(dev);
                 });

  return is_source_kernel_bundle_supported(Language, DeviceImplVec);
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
  std::string Source{std::string_view(SourceView)};
  include_pairs_t IncludePairs;
  size_t n = IncludePairViews.size();
  IncludePairs.reserve(n);
  for (auto &p : IncludePairViews)
    IncludePairs.push_back({ std::string{std::string_view(p.first)},
        std::string{std::string_view(p.second)}});

  if (!is_source_kernel_bundle_supported(Language, SyclContext))
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel_bundle creation from source not supported");

  // throw if include not supported?   awaiting guidance
  // if(!IncludePairs.empty() && is_include_supported(Languuage)){ throw invalid
  // }

  std::shared_ptr<kernel_bundle_impl> KBImpl =
      std::make_shared<kernel_bundle_impl>(SyclContext, Language, Source,
                                           IncludePairs);
  return sycl::detail::createSyclObjFromImpl<source_kb>(std::move(KBImpl));
}

source_kb make_kernel_bundle_from_source(const context &SyclContext,
                                         source_language Language,
                                         const std::vector<std::byte> &Bytes,
                                         include_pairs_view_t IncludePairs) {
  (void)IncludePairs;
  if (!is_source_kernel_bundle_supported(Language, SyclContext))
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel_bundle creation from source not supported");

  std::shared_ptr<kernel_bundle_impl> KBImpl =
      std::make_shared<kernel_bundle_impl>(SyclContext, Language, Bytes);
  return sycl::detail::createSyclObjFromImpl<source_kb>(std::move(KBImpl));
}

/////////////////////////
// syclex::detail::compile_from_source(source_kb) => obj_kb
/////////////////////////

obj_kb compile_from_source(
    source_kb &SourceKB, const std::vector<device> &Devices,
    const std::vector<sycl::detail::string_view> &BuildOptions,
    sycl::detail::string *LogView,
    const std::vector<sycl::detail::string_view> &RegisteredKernelNames) {
  std::string Log;
  std::string *LogPtr = nullptr;
  if (LogView)
    LogPtr = &Log;
  std::vector<device> UniqueDevices =
      sycl::detail::removeDuplicateDevices(Devices);
  std::shared_ptr<kernel_bundle_impl> sourceImpl = getSyclObjImpl(SourceKB);
  std::shared_ptr<kernel_bundle_impl> KBImpl = sourceImpl->compile_from_source(
      UniqueDevices, BuildOptions, LogPtr, RegisteredKernelNames);
  auto result = sycl::detail::createSyclObjFromImpl<obj_kb>(KBImpl);
  if (LogView)
    *LogView = Log;
  return result;
}

/////////////////////////
// syclex::detail::build_from_source(source_kb) => exe_kb
/////////////////////////

exe_kb build_from_source(
    source_kb &SourceKB, const std::vector<device> &Devices,
    const std::vector<sycl::detail::string_view> &BuildOptions,
    sycl::detail::string *LogView,
    const std::vector<sycl::detail::string_view> &RegisteredKernelNames) {
  std::string Log;
  std::string *LogPtr = nullptr;
  if (LogView)
    LogPtr = &Log;
  std::vector<device> UniqueDevices =
      sycl::detail::removeDuplicateDevices(Devices);
  const std::shared_ptr<kernel_bundle_impl> &sourceImpl =
      getSyclObjImpl(SourceKB);
  std::shared_ptr<kernel_bundle_impl> KBImpl = sourceImpl->build_from_source(
      UniqueDevices, BuildOptions, LogPtr, RegisteredKernelNames);
  auto result = sycl::detail::createSyclObjFromImpl<exe_kb>(std::move(KBImpl));
  if (LogView)
    *LogView = Log;
  return result;
}

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
