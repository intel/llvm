//==------- kernel_bundle_impl.hpp - SYCL kernel_bundle_impl ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_image_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <vector>

#include "split_string.hpp"

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental::detail {
using DeviceImplPtr = std::shared_ptr<sycl::detail::device_impl>;
bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const context &Ctx);

bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const std::vector<DeviceImplPtr> &Devices);
} // namespace ext::oneapi::experimental::detail

namespace detail {

static bool checkAllDevicesAreInContext(const std::vector<device> &Devices,
                                        const context &Context) {
  return std::all_of(
      Devices.begin(), Devices.end(), [&Context](const device &Dev) {
        return getSyclObjImpl(Context)->isDeviceValid(getSyclObjImpl(Dev));
      });
}

static bool checkAllDevicesHaveAspect(const std::vector<device> &Devices,
                                      aspect Aspect) {
  return std::all_of(Devices.begin(), Devices.end(),
                     [&Aspect](const device &Dev) { return Dev.has(Aspect); });
}

namespace syclex = sycl::ext::oneapi::experimental;

class kernel_impl;

/// The class is an impl counterpart of the sycl::kernel_bundle.
// It provides an access and utilities to manage set of sycl::device_images
// objects.
class kernel_bundle_impl {

  using SpecConstMapT = std::map<std::string, std::vector<unsigned char>>;

  void common_ctor_checks() const {
    const bool AllDevicesInTheContext =
        checkAllDevicesAreInContext(MDevices, MContext);
    if (MDevices.empty() || !AllDevicesInTheContext)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");

    if (bundle_state::input == MState &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_compiler))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_compiler");

    if (bundle_state::object == MState &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_linker))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_linker");
  }

public:
  kernel_bundle_impl(context Ctx, std::vector<device> Devs, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, State);
    fillUniqueDeviceImages();
  }

  // Interop constructor used by make_kernel
  kernel_bundle_impl(context Ctx, std::vector<device> Devs)
      : MContext(Ctx), MDevices(Devs), MState(bundle_state::executable) {
    if (!checkAllDevicesAreInContext(Devs, Ctx))
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");
  }

  // Interop constructor
  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     device_image_plain &DevImage)
      : kernel_bundle_impl(Ctx, Devs) {
    MDeviceImages.emplace_back(DevImage);
    MUniqueDeviceImages.emplace_back(DevImage);
  }

  // Matches sycl::build and sycl::compile
  // Have one constructor because sycl::build and sycl::compile have the same
  // signature
  kernel_bundle_impl(const kernel_bundle<bundle_state::input> &InputBundle,
                     std::vector<device> Devs, const property_list &PropList,
                     bundle_state TargetState)
      : MContext(InputBundle.get_context()), MDevices(std::move(Devs)),
        MState(TargetState) {

    const std::shared_ptr<kernel_bundle_impl> &InputBundleImpl =
        getSyclObjImpl(InputBundle);
    MSpecConstValues = InputBundleImpl->get_spec_const_map_ref();

    const std::vector<device> &InputBundleDevices =
        InputBundleImpl->get_devices();
    const bool AllDevsAssociatedWithInputBundle =
        std::all_of(MDevices.begin(), MDevices.end(),
                    [&InputBundleDevices](const device &Dev) {
                      return InputBundleDevices.end() !=
                             std::find(InputBundleDevices.begin(),
                                       InputBundleDevices.end(), Dev);
                    });
    if (MDevices.empty() || !AllDevsAssociatedWithInputBundle)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are in the set of associated "
          "devices for input bundle or vector of devices is empty");

    for (const DevImgPlainWithDeps &DevImgWithDeps :
         InputBundleImpl->MDeviceImages) {
      // Skip images which are not compatible with devices provided
      if (std::none_of(MDevices.begin(), MDevices.end(),
                       [&DevImgWithDeps](const device &Dev) {
                         return getSyclObjImpl(DevImgWithDeps.getMain())
                             ->compatible_with_device(Dev);
                       }))
        continue;

      switch (TargetState) {
      case bundle_state::object: {
        DevImgPlainWithDeps CompiledImgWithDeps =
            detail::ProgramManager::getInstance().compile(DevImgWithDeps,
                                                          MDevices, PropList);

        MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                   CompiledImgWithDeps.begin(),
                                   CompiledImgWithDeps.end());
        MDeviceImages.push_back(std::move(CompiledImgWithDeps));
        break;
      }

      case bundle_state::executable: {
        device_image_plain BuiltImg =
            detail::ProgramManager::getInstance().build(DevImgWithDeps,
                                                        MDevices, PropList);
        MDeviceImages.emplace_back(BuiltImg);
        MUniqueDeviceImages.emplace_back(BuiltImg);
        break;
      }
      case bundle_state::input:
      case bundle_state::ext_oneapi_source:
        throw exception(make_error_code(errc::runtime),
                        "Internal error. The target state should not be input "
                        "or ext_oneapi_source");
        break;
      }
    }
    removeDuplicateImages();
  }

  // Matches sycl::link
  kernel_bundle_impl(
      const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
      std::vector<device> Devs, const property_list &PropList)
      : MDevices(std::move(Devs)), MState(bundle_state::executable) {

    if (MDevices.empty())
      throw sycl::exception(make_error_code(errc::invalid),
                            "Vector of devices is empty");

    if (ObjectBundles.empty())
      return;

    MContext = ObjectBundles[0].get_context();
    for (size_t I = 1; I < ObjectBundles.size(); ++I) {
      if (ObjectBundles[I].get_context() != MContext)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same associated context");
    }

    // Check if any of the devices in devs are not in the set of associated
    // devices for any of the bundles in ObjectBundles
    const bool AllDevsAssociatedWithInputBundles = std::all_of(
        MDevices.begin(), MDevices.end(), [&ObjectBundles](const device &Dev) {
          // Number of devices is expected to be small
          return std::all_of(
              ObjectBundles.begin(), ObjectBundles.end(),
              [&Dev](const kernel_bundle<bundle_state::object> &KernelBundle) {
                const std::vector<device> &BundleDevices =
                    getSyclObjImpl(KernelBundle)->get_devices();
                return BundleDevices.end() != std::find(BundleDevices.begin(),
                                                        BundleDevices.end(),
                                                        Dev);
              });
        });
    if (!AllDevsAssociatedWithInputBundles)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices are in the set of associated "
                            "devices for input bundles");

    // TODO: Unify with c'tor for sycl::compile and sycl::build by calling
    // sycl::join on vector of kernel_bundles

    // The loop below just links each device image separately, not linking any
    // two device images together. This is correct so long as each device image
    // has no unresolved symbols. That's the case when device images are created
    // from generic SYCL APIs. There's no way in generic SYCL to create a kernel
    // which references an undefined symbol. If we decide in the future to allow
    // a backend interop API to create a "sycl::kernel_bundle" that references
    // undefined symbols, then the logic in this loop will need to be changed.
    for (const kernel_bundle<bundle_state::object> &ObjectBundle :
         ObjectBundles) {
      for (const DevImgPlainWithDeps &DeviceImageWithDeps :
           getSyclObjImpl(ObjectBundle)->MDeviceImages) {

        // Skip images which are not compatible with devices provided
        if (std::none_of(MDevices.begin(), MDevices.end(),
                         [&DeviceImageWithDeps](const device &Dev) {
                           return getSyclObjImpl(DeviceImageWithDeps.getMain())
                               ->compatible_with_device(Dev);
                         }))
          continue;

        std::vector<device_image_plain> LinkedResults =
            detail::ProgramManager::getInstance().link(DeviceImageWithDeps,
                                                       MDevices, PropList);
        MDeviceImages.insert(MDeviceImages.end(), LinkedResults.begin(),
                             LinkedResults.end());
        MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                   LinkedResults.begin(), LinkedResults.end());
      }
    }
    removeDuplicateImages();

    for (const kernel_bundle<bundle_state::object> &Bundle : ObjectBundles) {
      const KernelBundleImplPtr &BundlePtr = getSyclObjImpl(Bundle);
      for (const std::pair<const std::string, std::vector<unsigned char>>
               &SpecConst : BundlePtr->MSpecConstValues) {
        MSpecConstValues[SpecConst.first] = SpecConst.second;
      }
    }
  }

  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     const std::vector<kernel_id> &KernelIDs,
                     bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, KernelIDs, State);
    fillUniqueDeviceImages();
  }

  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     const DevImgSelectorImpl &Selector, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, Selector, State);
    fillUniqueDeviceImages();
  }

  // C'tor matches sycl::join API
  kernel_bundle_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles,
                     bundle_state State)
      : MState(State) {
    if (Bundles.empty())
      return;

    MContext = Bundles[0]->MContext;
    MDevices = Bundles[0]->MDevices;
    for (size_t I = 1; I < Bundles.size(); ++I) {
      if (Bundles[I]->MContext != MContext)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same associated context.");
      if (Bundles[I]->MDevices != MDevices)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same set of associated devices.");
    }

    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {
      MDeviceImages.insert(MDeviceImages.end(), Bundle->MDeviceImages.begin(),
                           Bundle->MDeviceImages.end());
      MSharedDeviceBinaries.insert(MSharedDeviceBinaries.end(),
                                   Bundle->MSharedDeviceBinaries.begin(),
                                   Bundle->MSharedDeviceBinaries.end());
    }

    fillUniqueDeviceImages();

    if (get_bundle_state() == bundle_state::input) {
      // Copy spec constants values from the device images.
      auto MergeSpecConstants = [this](const device_image_plain &Img) {
        const detail::DeviceImageImplPtr &ImgImpl = getSyclObjImpl(Img);
        const std::map<std::string,
                       std::vector<device_image_impl::SpecConstDescT>>
            &SpecConsts = ImgImpl->get_spec_const_data_ref();
        const std::vector<unsigned char> &Blob =
            ImgImpl->get_spec_const_blob_ref();
        for (const std::pair<const std::string,
                             std::vector<device_image_impl::SpecConstDescT>>
                 &SpecConst : SpecConsts) {
          if (SpecConst.second.front().IsSet)
            set_specialization_constant_raw_value(
                SpecConst.first.c_str(),
                Blob.data() + SpecConst.second.front().BlobOffset,
                SpecConst.second.back().CompositeOffset +
                    SpecConst.second.back().Size);
        }
      };
      std::for_each(begin(), end(), MergeSpecConstants);
    }

    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {
      for (const std::pair<const std::string, std::vector<unsigned char>>
               &SpecConst : Bundle->MSpecConstValues) {
        set_specialization_constant_raw_value(SpecConst.first.c_str(),
                                              SpecConst.second.data(),
                                              SpecConst.second.size());
      }
    }
  }

  // oneapi_ext_kernel_compiler
  // construct from source string
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::string &Src, include_pairs_t IncludePairsVec)
      : MContext(Context), MDevices(Context.get_devices()),
        MDeviceImages{device_image_plain{std::make_shared<device_image_impl>(
            Src, MContext, MDevices, Lang, std::move(IncludePairsVec))}},
        MUniqueDeviceImages{MDeviceImages[0].getMain()},
        MState(bundle_state::ext_oneapi_source) {
    common_ctor_checks();
  }

  // oneapi_ext_kernel_compiler
  // construct from source bytes
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::vector<std::byte> &Bytes)
      : MContext(Context), MDevices(Context.get_devices()),
        MDeviceImages{device_image_plain{std::make_shared<device_image_impl>(
            Bytes, MContext, MDevices, Lang)}},
        MUniqueDeviceImages{MDeviceImages[0].getMain()},
        MState(bundle_state::ext_oneapi_source) {
    common_ctor_checks();
  }

  // oneapi_ext_kernel_compiler
  // construct from built source files
  kernel_bundle_impl(
      const context &Context, const std::vector<device> &Devs,
      std::vector<device_image_plain> &&DevImgs,
      std::vector<std::shared_ptr<ManagedDeviceBinaries>> &&DevBinaries)
      : MContext(Context), MDevices(Devs),
        MSharedDeviceBinaries(std::move(DevBinaries)),
        MUniqueDeviceImages(std::move(DevImgs)),
        MState(bundle_state::executable) {
    common_ctor_checks();

    removeDuplicateImages();
    MDeviceImages.reserve(MUniqueDeviceImages.size());
    for (const device_image_plain &DevImg : MUniqueDeviceImages)
      MDeviceImages.emplace_back(DevImg);
  }

  std::shared_ptr<kernel_bundle_impl>
  build_from_source(const std::vector<device> Devices,
                    const std::vector<std::string> &BuildOptions,
                    std::string *LogPtr,
                    const std::vector<std::string> &RegisteredKernelNames) {
    assert(MState == bundle_state::ext_oneapi_source &&
           "bundle_state::ext_oneapi_source required");
    assert(allSourceBasedImages() && "All images must be source-based.");

    std::vector<device_image_plain> NewDevImgs;
    std::vector<std::shared_ptr<ManagedDeviceBinaries>> NewBinReso;
    for (device_image_plain &DevImg : MUniqueDeviceImages) {
      std::vector<std::shared_ptr<device_image_impl>> NewDevImgImpls =
          getSyclObjImpl(DevImg)->buildFromSource(
              Devices, BuildOptions, LogPtr, RegisteredKernelNames, NewBinReso);
      NewDevImgs.reserve(NewDevImgImpls.size());
      for (std::shared_ptr<device_image_impl> &DevImgImpl : NewDevImgImpls)
        NewDevImgs.emplace_back(std::move(DevImgImpl));
    }
    return std::make_shared<kernel_bundle_impl>(
        MContext, Devices, std::move(NewDevImgs), std::move(NewBinReso));
  }

public:
  bool ext_oneapi_has_kernel(const std::string &Name) const {
    return std::any_of(begin(), end(),
                       [&Name](const device_image_plain &DevImg) {
                         return getSyclObjImpl(DevImg)->hasKernelName(Name);
                       });
  }

  kernel
  ext_oneapi_get_kernel(const std::string &Name,
                        const std::shared_ptr<kernel_bundle_impl> &Self) const {
    if (!hasSourceBasedImages())
      throw sycl::exception(make_error_code(errc::invalid),
                            "'ext_oneapi_get_kernel' is only available in "
                            "kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state::ext_oneapi_source>.");

    // TODO: When linking is properly implemented for kernel compiler binaries,
    //       there can be scenarios where multiple binaries have the same
    //       kernels. In this case, all these bundles should be found and the
    //       resulting kernel object should be able to map devices to their
    //       respective backend kernel objects.
    for (const device_image_plain &DevImg : MUniqueDeviceImages) {
      const std::shared_ptr<device_image_impl> &DevImgImpl =
          getSyclObjImpl(DevImg);
      if (std::shared_ptr<kernel_impl> PotentialKernelImpl =
              DevImgImpl->tryGetSourceBasedKernel(Name, MContext, Self,
                                                  DevImgImpl))
        return detail::createSyclObjFromImpl<kernel>(
            std::move(PotentialKernelImpl));
    }
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel '" + Name + "' not found in kernel_bundle");
  }

  std::string ext_oneapi_get_raw_kernel_name(const std::string &Name) {
    if (!hasSourceBasedImages())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "'ext_oneapi_get_raw_kernel_name' is only available in "
          "kernel_bundles successfully built from "
          "kernel_bundle<bundle_state::ext_oneapi_source>.");

    auto It =
        std::find_if(begin(), end(), [&Name](const device_image_plain &DevImg) {
          return getSyclObjImpl(DevImg)->hasKernelName(Name);
        });
    if (It == end())
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel '" + Name + "' not found in kernel_bundle");

    return getSyclObjImpl(*It)->adjustKernelName(Name);
  }

  bool ext_oneapi_has_device_global(const std::string &Name) const {
    return std::any_of(
        begin(), end(), [&Name](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->hasDeviceGlobalName(Name);
        });
  }

  void *ext_oneapi_get_device_global_address(const std::string &Name,
                                             const device &Dev) const {
    DeviceGlobalMapEntry *Entry = getDeviceGlobalEntry(Name);

    if (std::find(MDevices.begin(), MDevices.end(), Dev) == MDevices.end()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel_bundle not built for device");
    }

    if (Entry->MIsDeviceImageScopeDecorated) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Cannot query USM pointer for device global with "
                            "'device_image_scope' property");
    }

    // TODO: Add context-only initialization via `urUSMContextMemcpyExp` instead
    // of using a throw-away queue.
    queue InitQueue{MContext, Dev};
    auto &USMMem =
        Entry->getOrAllocateDeviceGlobalUSM(getSyclObjImpl(InitQueue));
    InitQueue.wait_and_throw();
    return USMMem.getPtr();
  }

  size_t ext_oneapi_get_device_global_size(const std::string &Name) const {
    return getDeviceGlobalEntry(Name)->MDeviceGlobalTSize;
  }

  bool empty() const noexcept { return MDeviceImages.empty(); }

  backend get_backend() const noexcept {
    return MContext.get_platform().get_backend();
  }

  context get_context() const noexcept { return MContext; }

  const std::vector<device> &get_devices() const noexcept { return MDevices; }

  std::vector<kernel_id> get_kernel_ids() const {
    // Collect kernel ids from all device images, then remove duplicates
    std::vector<kernel_id> Result;
    for (const device_image_plain &DeviceImage : MUniqueDeviceImages) {
      const auto &DevImgImpl = getSyclObjImpl(DeviceImage);

      // RTC kernel bundles shouldn't have user-facing kernel ids, return an
      // empty vector when the bundle contains RTC kernels.
      if (DevImgImpl->getRTCInfo())
        continue;

      const std::vector<kernel_id> &KernelIDs = DevImgImpl->get_kernel_ids();

      Result.insert(Result.end(), KernelIDs.begin(), KernelIDs.end());
    }
    std::sort(Result.begin(), Result.end(), LessByNameComp{});

    auto NewIt = std::unique(Result.begin(), Result.end(), EqualByNameComp{});
    Result.erase(NewIt, Result.end());

    return Result;
  }

  kernel
  get_kernel(const kernel_id &KernelID,
             const std::shared_ptr<detail::kernel_bundle_impl> &Self) const {
    if (std::shared_ptr<kernel_impl> KernelImpl =
            tryGetOfflineKernel(KernelID, Self))
      return detail::createSyclObjFromImpl<kernel>(std::move(KernelImpl));
    throw sycl::exception(make_error_code(errc::invalid),
                          "The kernel bundle does not contain the kernel "
                          "identified by kernelId.");
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return std::any_of(begin(), end(),
                       [&KernelID](const device_image_plain &DeviceImage) {
                         return DeviceImage.has_kernel(KernelID);
                       });
  }

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return std::any_of(
        begin(), end(),
        [&KernelID, &Dev](const device_image_plain &DeviceImage) {
          return DeviceImage.has_kernel(KernelID, Dev);
        });
  }

  bool contains_specialization_constants() const noexcept {
    return std::any_of(
        begin(), end(), [](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->has_specialization_constants();
        });
  }

  bool native_specialization_constant() const noexcept {
    return contains_specialization_constants() &&
           std::all_of(begin(), end(),
                       [](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->all_specialization_constant_native();
                       });
  }

  bool has_specialization_constant(const char *SpecName) const noexcept {
    return std::any_of(begin(), end(),
                       [SpecName](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->has_specialization_constant(SpecName);
                       });
  }

  void set_specialization_constant_raw_value(const char *SpecName,
                                             const void *Value,
                                             size_t Size) noexcept {
    if (has_specialization_constant(SpecName))
      for (const device_image_plain &DeviceImage : MUniqueDeviceImages)
        getSyclObjImpl(DeviceImage)
            ->set_specialization_constant_raw_value(SpecName, Value);
    else {
      std::vector<unsigned char> &Val = MSpecConstValues[std::string{SpecName}];
      Val.resize(Size);
      std::memcpy(Val.data(), Value, Size);
    }
  }

  void get_specialization_constant_raw_value(const char *SpecName,
                                             void *ValueRet) const noexcept {
    for (const device_image_plain &DeviceImage : MUniqueDeviceImages)
      if (getSyclObjImpl(DeviceImage)->has_specialization_constant(SpecName)) {
        getSyclObjImpl(DeviceImage)
            ->get_specialization_constant_raw_value(SpecName, ValueRet);
        return;
      }

    // Specialization constant wasn't found in any of the device images,
    // try to fetch value from kernel_bundle.
    if (MSpecConstValues.count(std::string{SpecName}) != 0) {
      const std::vector<unsigned char> &Val =
          MSpecConstValues.at(std::string{SpecName});
      auto *Dest = static_cast<unsigned char *>(ValueRet);
      std::uninitialized_copy(Val.begin(), Val.end(), Dest);
      return;
    }

    assert(false &&
           "get_specialization_constant_raw_value called for missing constant");
  }

  bool is_specialization_constant_set(const char *SpecName) const noexcept {
    bool SetInDevImg = std::any_of(
        begin(), end(), [SpecName](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)
              ->is_specialization_constant_set(SpecName);
        });
    return SetInDevImg || MSpecConstValues.count(std::string{SpecName}) != 0;
  }

  const device_image_plain *begin() const { return MUniqueDeviceImages.data(); }

  const device_image_plain *end() const {
    return MUniqueDeviceImages.data() + MUniqueDeviceImages.size();
  }

  size_t size() const noexcept { return MUniqueDeviceImages.size(); }

  bundle_state get_bundle_state() const { return MState; }

  const SpecConstMapT &get_spec_const_map_ref() const noexcept {
    return MSpecConstValues;
  }

  bool add_kernel(const kernel_id &KernelID, const device &Dev) {
    // Skip if kernel is already there
    if (has_kernel(KernelID, Dev))
      return true;

    // First try and get images in current bundle state
    const bundle_state BundleState = get_bundle_state();
    std::vector<DevImgPlainWithDeps> NewDevImgs =
        detail::ProgramManager::getInstance().getSYCLDeviceImages(
            MContext, {Dev}, {KernelID}, BundleState);

    // No images found so we report as not inserted
    if (NewDevImgs.empty())
      return false;

    // Propagate already set specialization constants to the new images
    for (DevImgPlainWithDeps &DevImgWithDeps : NewDevImgs)
      for (device_image_plain &DevImg : DevImgWithDeps)
        for (auto SpecConst : MSpecConstValues)
          getSyclObjImpl(DevImg)->set_specialization_constant_raw_value(
              SpecConst.first.c_str(), SpecConst.second.data());

    // Add the images to the collection
    MDeviceImages.insert(MDeviceImages.end(), NewDevImgs.begin(),
                         NewDevImgs.end());
    removeDuplicateImages();
    return true;
  }

  bool hasSourceBasedImages() const noexcept {
    return std::any_of(begin(), end(), [](const device_image_plain &DevImg) {
      return getSyclObjImpl(DevImg)->getOriginMask() &
             ImageOriginKernelCompiler;
    });
  }

  bool hasSYCLOfflineImages() const noexcept {
    return std::any_of(begin(), end(), [](const device_image_plain &DevImg) {
      return getSyclObjImpl(DevImg)->getOriginMask() & ImageOriginSYCLOffline;
    });
  }

  bool allSourceBasedImages() const noexcept {
    return std::all_of(begin(), end(), [](const device_image_plain &DevImg) {
      return getSyclObjImpl(DevImg)->getOriginMask() &
             ImageOriginKernelCompiler;
    });
  }

  std::shared_ptr<kernel_impl> tryGetOfflineKernel(
      const kernel_id &KernelID,
      const std::shared_ptr<detail::kernel_bundle_impl> &Self) const {
    using ImageImpl = std::shared_ptr<detail::device_image_impl>;
    // Selected image.
    ImageImpl SelectedImage = nullptr;
    // Image where specialization constants are replaced with default values.
    ImageImpl ImageWithReplacedSpecConsts = nullptr;
    // Original image where specialization constants are not replaced with
    // default values.
    ImageImpl OriginalImage = nullptr;
    // Used to track if any of the candidate images has specialization values
    // set.
    bool SpecConstsSet = false;
    for (const DevImgPlainWithDeps &DeviceImageWithDeps : MDeviceImages) {
      const device_image_plain &DeviceImage = DeviceImageWithDeps.getMain();
      if (!DeviceImageWithDeps.getMain().has_kernel(KernelID))
        continue;

      const auto DeviceImageImpl = detail::getSyclObjImpl(DeviceImage);
      SpecConstsSet |= DeviceImageImpl->is_any_specialization_constant_set();

      // Remember current image in corresponding variable depending on whether
      // specialization constants are replaced with default value or not.
      (DeviceImageImpl->specialization_constants_replaced_with_default()
           ? ImageWithReplacedSpecConsts
           : OriginalImage) = DeviceImageImpl;

      if (SpecConstsSet) {
        // If specialization constant is set in any of the candidate images
        // then we can't use ReplacedImage, so we select NativeImage if any or
        // we select OriginalImage and keep iterating in case there is an image
        // with native support.
        SelectedImage = OriginalImage;
        if (SelectedImage &&
            SelectedImage->all_specialization_constant_native())
          break;
      } else {
        // For now select ReplacedImage but it may be reset if any of the
        // further device images has specialization constant value set. If after
        // all iterations specialization constant values are not set in any of
        // the candidate images then that will be the selected image.
        // Also we don't want to use ReplacedImage if device image has native
        // support.
        if (ImageWithReplacedSpecConsts &&
            !ImageWithReplacedSpecConsts->all_specialization_constant_native())
          SelectedImage = ImageWithReplacedSpecConsts;
        else
          // In case if we don't have or don't use ReplacedImage.
          SelectedImage = OriginalImage;
      }
    }

    if (!SelectedImage)
      return nullptr;

    auto [Kernel, CacheMutex, ArgMask] =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            MContext, KernelID.get_name(), /*PropList=*/{},
            SelectedImage->get_ur_program_ref());

    return std::make_shared<kernel_impl>(
        Kernel, detail::getSyclObjImpl(MContext), SelectedImage, Self, ArgMask,
        SelectedImage->get_ur_program_ref(), CacheMutex);
  }

  std::shared_ptr<kernel_impl>
  tryGetKernel(const std::string &Name,
               const std::shared_ptr<kernel_bundle_impl> &Self) const {
    // TODO: For source-based kernels, it may be faster to keep a map between
    //       {kernel_name, device} and their corresponding image.
    // First look through the kernels registered in source-based images.
    for (const device_image_plain &DevImg : MUniqueDeviceImages) {
      const std::shared_ptr<device_image_impl> &DevImgImpl =
          getSyclObjImpl(DevImg);
      if (std::shared_ptr<kernel_impl> SourceBasedKernel =
              DevImgImpl->tryGetSourceBasedKernel(Name, MContext, Self,
                                                  DevImgImpl))
        return SourceBasedKernel;
    }

    // Fall back to regular offline compiled kernel_bundle look-up.
    if (std::optional<kernel_id> MaybeKernelID =
            sycl::detail::ProgramManager::getInstance().tryGetSYCLKernelID(
                Name))
      return tryGetOfflineKernel(*MaybeKernelID, Self);
    return nullptr;
  }

private:
  DeviceGlobalMapEntry *getDeviceGlobalEntry(const std::string &Name) const {
    if (!hasSourceBasedImages()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Querying device globals by name is only available "
                            "in kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state>::ext_oneapi_source> "
                            "with 'sycl' source language.");
    }

    if (!ext_oneapi_has_device_global(Name)) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "device global '" + Name +
                                "' not found in kernel_bundle");
    }

    for (const device_image_plain &DevImg : MUniqueDeviceImages)
      if (DeviceGlobalMapEntry *Entry =
              getSyclObjImpl(DevImg)->tryGetDeviceGlobalEntry(Name))
        return Entry;
    assert(false && "Device global should have been found.");
    return nullptr;
  }

  void fillUniqueDeviceImages() {
    assert(MUniqueDeviceImages.empty());
    for (const DevImgPlainWithDeps &Imgs : MDeviceImages)
      MUniqueDeviceImages.insert(MUniqueDeviceImages.end(), Imgs.begin(),
                                 Imgs.end());
    removeDuplicateImages();
  }
  void removeDuplicateImages() {
    std::sort(MUniqueDeviceImages.begin(), MUniqueDeviceImages.end(),
              LessByHash<device_image_plain>{});
    const auto It =
        std::unique(MUniqueDeviceImages.begin(), MUniqueDeviceImages.end());
    MUniqueDeviceImages.erase(It, MUniqueDeviceImages.end());
  }

  context MContext;
  std::vector<device> MDevices;

  // For sycl_jit, building from source may have produced sycl binaries that
  // the kernel_bundles now manage.
  // NOTE: This must appear before device images to enforce their freeing of
  //       device globals prior to unregistering the binaries.
  std::vector<std::shared_ptr<ManagedDeviceBinaries>> MSharedDeviceBinaries;

  std::vector<DevImgPlainWithDeps> MDeviceImages;
  std::vector<device_image_plain> MUniqueDeviceImages;
  // This map stores values for specialization constants, that are missing
  // from any device image.
  SpecConstMapT MSpecConstValues;
  bundle_state MState;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
