//==------- kernel_bundle_impl.hpp - SYCL kernel_bundle_impl ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_image_impl.hpp>
#include <detail/kernel_compiler/kernel_compiler_opencl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/common_info.hpp>
#include <sycl/detail/pi.h>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
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

  void common_ctor_checks(bundle_state State) {
    const bool AllDevicesInTheContext =
        checkAllDevicesAreInContext(MDevices, MContext);
    if (MDevices.empty() || !AllDevicesInTheContext)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");

    if (bundle_state::input == State &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_compiler))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_compiler");

    if (bundle_state::object == State &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_linker))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_linker");
  }

public:
  kernel_bundle_impl(context Ctx, std::vector<device> Devs, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks(State);

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, State);
  }

  // Interop constructor used by make_kernel
  kernel_bundle_impl(context Ctx, std::vector<device> Devs)
      : MContext(Ctx), MDevices(Devs), MState(bundle_state::executable) {
    if (!checkAllDevicesAreInContext(Devs, Ctx))
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");
    MIsInterop = true;
  }

  // Interop constructor
  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     device_image_plain &DevImage)
      : kernel_bundle_impl(Ctx, Devs) {
    MDeviceImages.push_back(DevImage);
  }

  // Matches sycl::build and sycl::compile
  // Have one constructor because sycl::build and sycl::compile have the same
  // signature
  kernel_bundle_impl(const kernel_bundle<bundle_state::input> &InputBundle,
                     std::vector<device> Devs, const property_list &PropList,
                     bundle_state TargetState)
      : MContext(InputBundle.get_context()), MDevices(std::move(Devs)),
        MState(TargetState) {

    MSpecConstValues = getSyclObjImpl(InputBundle)->get_spec_const_map_ref();

    const std::vector<device> &InputBundleDevices =
        getSyclObjImpl(InputBundle)->get_devices();
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

    for (const device_image_plain &DeviceImage : InputBundle) {
      // Skip images which are not compatible with devices provided
      if (std::none_of(
              MDevices.begin(), MDevices.end(),
              [&DeviceImage](const device &Dev) {
                return getSyclObjImpl(DeviceImage)->compatible_with_device(Dev);
              }))
        continue;

      switch (TargetState) {
      case bundle_state::object:
        MDeviceImages.push_back(detail::ProgramManager::getInstance().compile(
            DeviceImage, MDevices, PropList));
        break;
      case bundle_state::executable:
        MDeviceImages.push_back(detail::ProgramManager::getInstance().build(
            DeviceImage, MDevices, PropList));
        break;
      case bundle_state::input:
      case bundle_state::ext_oneapi_source:
        throw sycl::runtime_error("Internal error. The target state should not "
                                  "be input or ext_oneapi_source",
                                  PI_ERROR_INVALID_OPERATION);
        break;
      }
    }
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

    // TODO: Unify with c'tor for sycl::comile and sycl::build by calling
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
      for (const device_image_plain &DeviceImage : ObjectBundle) {

        // Skip images which are not compatible with devices provided
        if (std::none_of(MDevices.begin(), MDevices.end(),
                         [&DeviceImage](const device &Dev) {
                           return getSyclObjImpl(DeviceImage)
                               ->compatible_with_device(Dev);
                         }))
          continue;

        std::vector<device_image_plain> LinkedResults =
            detail::ProgramManager::getInstance().link(DeviceImage, MDevices,
                                                       PropList);
        MDeviceImages.insert(MDeviceImages.end(), LinkedResults.begin(),
                             LinkedResults.end());
      }
    }

    for (const kernel_bundle<bundle_state::object> &Bundle : ObjectBundles) {
      const KernelBundleImplPtr BundlePtr = getSyclObjImpl(Bundle);
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

    common_ctor_checks(State);

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, KernelIDs, State);
  }

  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     const DevImgSelectorImpl &Selector, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks(State);

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, Selector, State);
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
    }

    std::sort(MDeviceImages.begin(), MDeviceImages.end(),
              LessByHash<device_image_plain>{});

    if (get_bundle_state() == bundle_state::input) {
      // Copy spec constants values from the device images to be removed.
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
      std::for_each(MDeviceImages.begin(), MDeviceImages.end(),
                    MergeSpecConstants);
    }

    const auto DevImgIt =
        std::unique(MDeviceImages.begin(), MDeviceImages.end());

    // Remove duplicate device images.
    MDeviceImages.erase(DevImgIt, MDeviceImages.end());

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
                     const std::string &Src)
      : MContext(Context), MDevices(Context.get_devices()),
        MState(bundle_state::ext_oneapi_source), Language(Lang), Source(Src) {}

  // oneapi_ext_kernel_compiler
  // construct from source bytes
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::vector<std::byte> &Bytes)
      : MContext(Context), MDevices(Context.get_devices()),
        MState(bundle_state::ext_oneapi_source), Language(Lang), Source(Bytes) {
  }

  // oneapi_ext_kernel_compiler
  // interop constructor
  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     device_image_plain &DevImage,
                     std::vector<std::string> KNames)
      : kernel_bundle_impl(Ctx, Devs, DevImage) {
    MState = bundle_state::executable;
    KernelNames = KNames;
  }

  std::shared_ptr<kernel_bundle_impl>
  build_from_source(const std::vector<device> Devices,
                    const std::vector<std::string> &BuildOptions,
                    std::string *LogPtr) {
    assert(MState == bundle_state::ext_oneapi_source &&
           "bundle_state::ext_oneapi_source required");

    using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
    ContextImplPtr ContextImpl = getSyclObjImpl(MContext);
    const PluginPtr &Plugin = ContextImpl->getPlugin();

    std::vector<pi::PiDevice> DeviceVec;
    DeviceVec.reserve(Devices.size());
    for (const auto &SyclDev : Devices) {
      pi::PiDevice Dev = getSyclObjImpl(SyclDev)->getHandleRef();
      DeviceVec.push_back(Dev);
    }

    const auto spirv = [&]() -> std::vector<uint8_t> {
      if (Language == syclex::source_language::opencl) {
        // if successful, the log is empty. if failed, throws an error with the
        // compilation log.
        const auto &SourceStr = std::get<std::string>(this->Source);
        std::vector<uint32_t> IPVersionVec(Devices.size());
        std::transform(DeviceVec.begin(), DeviceVec.end(), IPVersionVec.begin(),
                       [&](pi::PiDevice d) {
                         uint32_t ipVersion = 0;
                         Plugin->call<PiApiKind::piDeviceGetInfo>(
                             d, PI_EXT_ONEAPI_DEVICE_INFO_IP_VERSION,
                             sizeof(uint32_t), &ipVersion, nullptr);
                         return ipVersion;
                       });
        return syclex::detail::OpenCLC_to_SPIRV(SourceStr, IPVersionVec,
                                                BuildOptions, LogPtr);
      }
      if (Language == syclex::source_language::spirv) {
        const auto &SourceBytes =
            std::get<std::vector<std::byte>>(this->Source);
        std::vector<uint8_t> Result(SourceBytes.size());
        std::transform(SourceBytes.cbegin(), SourceBytes.cend(), Result.begin(),
                       [](std::byte B) { return static_cast<uint8_t>(B); });
        return Result;
      }
      throw sycl::exception(
          make_error_code(errc::invalid),
          "OpenCL C and SPIR-V are the only supported languages at this time");
    }();

    sycl::detail::pi::PiProgram PiProgram = nullptr;
    Plugin->call<PiApiKind::piProgramCreate>(
        ContextImpl->getHandleRef(), spirv.data(), spirv.size(), &PiProgram);
    // program created by piProgramCreate is implicitly retained.

    Plugin->call<errc::build, PiApiKind::piProgramBuild>(
        PiProgram, DeviceVec.size(), DeviceVec.data(), nullptr, nullptr,
        nullptr);

    // Get the number of kernels in the program.
    size_t NumKernels;
    Plugin->call<PiApiKind::piProgramGetInfo>(
        PiProgram, PI_PROGRAM_INFO_NUM_KERNELS, sizeof(size_t), &NumKernels,
        nullptr);

    // Get the kernel names.
    size_t KernelNamesSize;
    Plugin->call<PiApiKind::piProgramGetInfo>(
        PiProgram, PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr, &KernelNamesSize);

    // semi-colon delimited list of kernel names.
    std::string KernelNamesStr(KernelNamesSize, ' ');
    Plugin->call<PiApiKind::piProgramGetInfo>(
        PiProgram, PI_PROGRAM_INFO_KERNEL_NAMES, KernelNamesStr.size(),
        &KernelNamesStr[0], nullptr);
    std::vector<std::string> KernelNames =
        detail::split_string(KernelNamesStr, ';');

    // make the device image and the kernel_bundle_impl
    auto KernelIDs = std::make_shared<std::vector<kernel_id>>();
    auto DevImgImpl = std::make_shared<device_image_impl>(
        nullptr, MContext, MDevices, bundle_state::executable, KernelIDs,
        PiProgram);
    device_image_plain DevImg{DevImgImpl};
    return std::make_shared<kernel_bundle_impl>(MContext, MDevices, DevImg,
                                                KernelNames);
  }

  bool ext_oneapi_has_kernel(const std::string &Name) {
    auto it = std::find(KernelNames.begin(), KernelNames.end(), Name);
    return it != KernelNames.end();
  }

  kernel
  ext_oneapi_get_kernel(const std::string &Name,
                        const std::shared_ptr<kernel_bundle_impl> &Self) {
    if (KernelNames.empty())
      throw sycl::exception(make_error_code(errc::invalid),
                            "'ext_oneapi_get_kernel' is only available in "
                            "kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state:ext_oneapi_source>.");

    if (!ext_oneapi_has_kernel(Name))
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel '" + Name + "' not found in kernel_bundle");

    assert(MDeviceImages.size() > 0);
    const std::shared_ptr<detail::device_image_impl> &DeviceImageImpl =
        detail::getSyclObjImpl(MDeviceImages[0]);
    sycl::detail::pi::PiProgram PiProgram = DeviceImageImpl->get_program_ref();
    ContextImplPtr ContextImpl = getSyclObjImpl(MContext);
    const PluginPtr &Plugin = ContextImpl->getPlugin();
    sycl::detail::pi::PiKernel PiKernel = nullptr;
    Plugin->call<PiApiKind::piKernelCreate>(PiProgram, Name.c_str(), &PiKernel);
    // Kernel created by piKernelCreate is implicitly retained.

    std::shared_ptr<kernel_impl> KernelImpl = std::make_shared<kernel_impl>(
        PiKernel, detail::getSyclObjImpl(MContext), Self);

    return detail::createSyclObjFromImpl<kernel>(KernelImpl);
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
    for (const device_image_plain &DeviceImage : MDeviceImages) {
      const std::vector<kernel_id> &KernelIDs =
          getSyclObjImpl(DeviceImage)->get_kernel_ids();

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
    for (auto &DeviceImage : MDeviceImages) {
      if (!DeviceImage.has_kernel(KernelID))
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
      throw sycl::exception(make_error_code(errc::invalid),
                            "The kernel bundle does not contain the kernel "
                            "identified by kernelId.");

    auto [Kernel, CacheMutex, ArgMask] =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            MContext, KernelID.get_name(), /*PropList=*/{},
            SelectedImage->get_program_ref());

    std::shared_ptr<kernel_impl> KernelImpl =
        std::make_shared<kernel_impl>(Kernel, detail::getSyclObjImpl(MContext),
                                      SelectedImage, Self, ArgMask, CacheMutex);

    return detail::createSyclObjFromImpl<kernel>(KernelImpl);
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return std::any_of(MDeviceImages.begin(), MDeviceImages.end(),
                       [&KernelID](const device_image_plain &DeviceImage) {
                         return DeviceImage.has_kernel(KernelID);
                       });
  }

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return std::any_of(
        MDeviceImages.begin(), MDeviceImages.end(),
        [&KernelID, &Dev](const device_image_plain &DeviceImage) {
          return DeviceImage.has_kernel(KernelID, Dev);
        });
  }

  bool contains_specialization_constants() const noexcept {
    return std::any_of(
        MDeviceImages.begin(), MDeviceImages.end(),
        [](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->has_specialization_constants();
        });
  }

  bool native_specialization_constant() const noexcept {
    return contains_specialization_constants() &&
           std::all_of(MDeviceImages.begin(), MDeviceImages.end(),
                       [](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->all_specialization_constant_native();
                       });
  }

  bool has_specialization_constant(const char *SpecName) const noexcept {
    return std::any_of(MDeviceImages.begin(), MDeviceImages.end(),
                       [SpecName](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->has_specialization_constant(SpecName);
                       });
  }

  void set_specialization_constant_raw_value(const char *SpecName,
                                             const void *Value,
                                             size_t Size) noexcept {
    if (has_specialization_constant(SpecName))
      for (const device_image_plain &DeviceImage : MDeviceImages)
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
    for (const device_image_plain &DeviceImage : MDeviceImages)
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
    bool SetInDevImg =
        std::any_of(MDeviceImages.begin(), MDeviceImages.end(),
                    [SpecName](const device_image_plain &DeviceImage) {
                      return getSyclObjImpl(DeviceImage)
                          ->is_specialization_constant_set(SpecName);
                    });
    return SetInDevImg || MSpecConstValues.count(std::string{SpecName}) != 0;
  }

  const device_image_plain *begin() const { return MDeviceImages.data(); }

  const device_image_plain *end() const {
    return MDeviceImages.data() + MDeviceImages.size();
  }

  size_t size() const noexcept { return MDeviceImages.size(); }

  bundle_state get_bundle_state() const { return MState; }

  const SpecConstMapT &get_spec_const_map_ref() const noexcept {
    return MSpecConstValues;
  }

  bool isInterop() const { return MIsInterop; }

  bool add_kernel(const kernel_id &KernelID, const device &Dev) {
    // Skip if kernel is already there
    if (has_kernel(KernelID, Dev))
      return true;

    // First try and get images in current bundle state
    const bundle_state BundleState = get_bundle_state();
    std::vector<device_image_plain> NewDevImgs =
        detail::ProgramManager::getInstance().getSYCLDeviceImages(
            MContext, {Dev}, {KernelID}, BundleState);

    // No images found so we report as not inserted
    if (NewDevImgs.empty())
      return false;

    // Propagate already set specialization constants to the new images
    for (device_image_plain &DevImg : NewDevImgs)
      for (auto SpecConst : MSpecConstValues)
        getSyclObjImpl(DevImg)->set_specialization_constant_raw_value(
            SpecConst.first.c_str(), SpecConst.second.data());

    // Add the images to the collection
    MDeviceImages.insert(MDeviceImages.end(), NewDevImgs.begin(),
                         NewDevImgs.end());
    return true;
  }

private:
  context MContext;
  std::vector<device> MDevices;
  std::vector<device_image_plain> MDeviceImages;
  // This map stores values for specialization constants, that are missing
  // from any device image.
  SpecConstMapT MSpecConstValues;
  bool MIsInterop = false;
  bundle_state MState;
  // ext_oneapi_kernel_compiler : Source, Languauge, KernelNames
  const syclex::source_language Language = syclex::source_language::opencl;
  const std::variant<std::string, std::vector<std::byte>> Source;
  // only kernel_bundles created from source have KernelNames member.
  std::vector<std::string> KernelNames;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
