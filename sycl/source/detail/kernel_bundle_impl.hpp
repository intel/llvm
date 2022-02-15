//==------- kernel_bundle_impl.hpp - SYCL kernel_bundle_impl ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <class T> struct LessByHash {
  bool operator()(const T &LHS, const T &RHS) const {
    return getSyclObjImpl(LHS) < getSyclObjImpl(RHS);
  }
};

static bool checkAllDevicesAreInContext(const std::vector<device> &Devices,
                                        const context &Context) {
  const std::vector<device> &ContextDevices = Context.get_devices();
  return std::all_of(
      Devices.begin(), Devices.end(), [&ContextDevices](const device &Dev) {
        return ContextDevices.end() !=
               std::find(ContextDevices.begin(), ContextDevices.end(), Dev);
      });
}

static bool checkAllDevicesHaveAspect(const std::vector<device> &Devices,
                                      aspect Aspect) {
  return std::all_of(Devices.begin(), Devices.end(),
                     [&Aspect](const device &Dev) { return Dev.has(Aspect); });
}

// The class is an impl counterpart of the sycl::kernel_bundle.
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
        throw sycl::runtime_error(
            "Internal error. The target state should not be input",
            PI_INVALID_OPERATION);
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

        const std::vector<device_image_plain> VectorOfOneImage{DeviceImage};
        std::vector<device_image_plain> LinkedResults =
            detail::ProgramManager::getInstance().link(VectorOfOneImage,
                                                       MDevices, PropList);
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

    // TODO: Add a check that all kernel ids are compatible with at least one
    // device in Devs
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

    auto It = std::find_if(MDeviceImages.begin(), MDeviceImages.end(),
                           [&KernelID](const device_image_plain &DeviceImage) {
                             return DeviceImage.has_kernel(KernelID);
                           });

    if (MDeviceImages.end() == It)
      throw sycl::exception(make_error_code(errc::invalid),
                            "The kernel bundle does not contain the kernel "
                            "identified by kernelId.");

    const std::shared_ptr<detail::device_image_impl> &DeviceImageImpl =
        detail::getSyclObjImpl(*It);

    RT::PiKernel Kernel = nullptr;
    std::tie(Kernel, std::ignore) =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            MContext, KernelID.get_name(), /*PropList=*/{},
            DeviceImageImpl->get_program_ref());

    std::shared_ptr<kernel_impl> KernelImpl = std::make_shared<kernel_impl>(
        Kernel, detail::getSyclObjImpl(MContext), DeviceImageImpl, Self);

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
    return std::all_of(MDeviceImages.begin(), MDeviceImages.end(),
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
      const auto *DataPtr = static_cast<const unsigned char *>(Value);
      std::vector<unsigned char> &Val = MSpecConstValues[std::string{SpecName}];
      Val.resize(Size);
      Val.insert(Val.begin(), DataPtr, DataPtr + Size);
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
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
