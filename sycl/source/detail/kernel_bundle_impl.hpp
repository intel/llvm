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
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <detail/device_image_impl.hpp>
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
  bool operator()(const T &LHS, const T &RHS) {
    return getSyclObjImpl(LHS) < getSyclObjImpl(RHS);
  }
};

// The class is an impl counterpart of the sycl::kernel_bundle.
// It provides an access and utilities to manage set of sycl::device_images
// objects.
class kernel_bundle_impl {

public:
  kernel_bundle_impl(context Ctx, std::vector<device> Devs, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)) {

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, State);
  }

  // Matches sycl::build
  kernel_bundle_impl(const kernel_bundle<bundle_state::input> &InputBundle,
                     const std::vector<device> &Devs,
                     const property_list &PropList) {

    // TODO: Add checks here

    for (const device_image_plain &DeviceImage : InputBundle) {
      MDeviceImages.push_back(detail::ProgramManager::getInstance().build(
          DeviceImage, Devs, PropList));
    }
  }

  // Matches sycl::compile
  // TODO: Replace kernel_bunlde_impl with kernel_bundle_plain
  kernel_bundle_impl(const std::shared_ptr<kernel_bundle_impl> &InputBundleImpl,
                     const std::vector<device> Devs,
                     const property_list &PropList)
      : MContext(InputBundleImpl->MContext), MDevices(std::move(Devs)) {

    for (const device_image_plain &DeviceImage : *InputBundleImpl) {
      if (std::none_of(
              MDevices.begin(), MDevices.end(),
              [&DeviceImage](const device &Dev) {
                return getSyclObjImpl(DeviceImage)->compatible_with_device(Dev);
              }))
        continue;

      MDeviceImages.push_back(detail::ProgramManager::getInstance().compile(
          DeviceImage, Devs, PropList));
    }
  }

  // Matches sycl::link
  kernel_bundle_impl(
      const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
      std::vector<device> Devs, const property_list &PropList)
      : MContext(ObjectBundles[0].get_context()), MDevices(std::move(Devs)) {

    for(const device &Dev: Devs) {
      for(const kernel_bundle<bundle_state::object> &ObjectBundle: ObjectBundles) {
        const std::vector<device> &BundleDevices =
            getSyclObjImpl(ObjectBundle)->MDevices;

        if (std::none_of(
                BundleDevices.begin(), BundleDevices.end(),
                [&Dev](const device &DevCand) { return Dev == DevCand; }))
          throw "laga";
      }
    }

    std::vector<device_image_plain> DeviceImages;
    for (const kernel_bundle<bundle_state::object> &ObjectBundle :
         ObjectBundles) {
      DeviceImages.insert(DeviceImages.end(), ObjectBundle.begin(),
                          ObjectBundle.end());
    }

    MDeviceImages = detail::ProgramManager::getInstance().link(DeviceImages,
                                                               Devs, PropList);
  }

  kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                     const std::vector<kernel_id> &KernelIDs,
                     bundle_state State)
      : kernel_bundle_impl(Ctx, Devs, State) {

    // Filter out images that have no kernel_ids specified
    auto It = std::remove_if(MDeviceImages.begin(), MDeviceImages.end(),
                             [&KernelIDs](const device_image_plain &Image) {
                               return std::none_of(
                                   KernelIDs.begin(), KernelIDs.end(),
                                   [&Image](const sycl::kernel_id &KernelID) {
                                     return Image.has_kernel(KernelID);
                                   });
                             });
    MDeviceImages.erase(It, MDeviceImages.end());
  }

  kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                     const DevImgSelectorImpl &Selector, bundle_state State)
      : kernel_bundle_impl(Ctx, Devs, State) {

    // Filter out images that are rejected by Selector
    auto It = std::remove_if(MDeviceImages.begin(), MDeviceImages.end(),
                             [&Selector](const device_image_plain &Image) {
                               return !Selector(getSyclObjImpl(Image));
                             });
    MDeviceImages.erase(It, MDeviceImages.end());
  }

  // C'tor matches sycl::join API
  kernel_bundle_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles) {
    MContext = Bundles[0]->MContext;
    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {

      // Insert devices from a bundle keeping MDevices sorted.
      const size_t DevCurSize = MDevices.size();
      MDevices.insert(MDevices.end(), Bundle->MDevices.begin(),
                      Bundle->MDevices.end());
      std::inplace_merge(MDevices.begin(), MDevices.begin() + DevCurSize,
                         MDevices.end(), LessByHash<device>{});

      // Insert devices from a bundle keeping MDeviceImages sorted.
      const size_t ImgCurSize = MDeviceImages.size();
      MDeviceImages.insert(MDeviceImages.end(), Bundle->MDeviceImages.begin(),
                           Bundle->MDeviceImages.end());

      std::inplace_merge(MDeviceImages.begin(),
                         MDeviceImages.begin() + ImgCurSize,
                         MDeviceImages.end(), LessByHash<device_image_plain>{});
    }

    const auto DevIt = std::unique(MDevices.begin(), MDevices.end());
    MDevices.erase(DevIt, MDevices.end());

    const auto DevImgIt =
        std::unique(MDeviceImages.begin(), MDeviceImages.end());
    MDeviceImages.erase(DevImgIt, MDeviceImages.end());
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

    auto NewIt =
        std::unique(Result.begin(), Result.end(),
                    [](const sycl::kernel_id &LHS, const sycl::kernel_id &RHS) {
                      return strcmp(LHS.get_name(), RHS.get_name()) == 0;
                    }

        );

    //std::sort(Result.begin(), Result.end(), LessByHash<kernel_id>{});
    Result.erase(NewIt, Result.end());

    return Result;
  }

  kernel get_kernel(const kernel_id &KernelID) const {
    (void)KernelID;
    //auto It = std::find_if(MDeviceImages.begin(), MDeviceImages.end(),
                           //[&KernelID](const device_image_plain &DeviceImage) {
                             //return DeviceImage.has_kernel(KernelID);
                           //});
    //const std::shared_ptr<detail::device_image_impl> &DeviceImageImpl =
        //detail::getSyclObjImpl(*It);
    //const device &Dev = DeviceImageImpl->get_devices()[0];
    //RT::PiKernel Kernel =
        //detail::ProgramManager::getInstance().getOrCreateKernel(
            //(-1), MContext, Dev, KernelID.get_name(), {});

    //std::shared_ptr<kernel_impl> KernelImpl =
        //std::make_shared<kernel_impl>(Kernel, Self);

    //return detail::createSyclObjFromImpl(KernelImpl);

    throw sycl::runtime_error("Not implemented", PI_INVALID_OPERATION);
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

  bool has_specialization_constant(unsigned int SpecID) const noexcept {
    return std::any_of(MDeviceImages.begin(), MDeviceImages.end(),
                       [SpecID](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->has_specialization_constant(SpecID);
                       });
  }

  void set_specialization_constant_raw_value(unsigned int SpecID,
                                             const void *Value,
                                             size_t ValueSize) {
    for (const device_image_plain &DeviceImage : MDeviceImages)
      getSyclObjImpl(DeviceImage)
          ->set_specialization_constant_raw_value(SpecID, Value, ValueSize);
  }

  const void *get_specialization_constant_raw_value(unsigned int SpecID,
                                                    void *ValueRet,
                                                    size_t ValueSize) const {
    for (const device_image_plain &DeviceImage : MDeviceImages)
      if (getSyclObjImpl(DeviceImage)->has_specialization_constant(SpecID)) {
        getSyclObjImpl(DeviceImage)
            ->get_specialization_constant_raw_value(SpecID, ValueRet,
                                                    ValueSize);
      }

    return nullptr;
  }

  const device_image_plain *begin() const { return &MDeviceImages.front(); }

  const device_image_plain *end() const { return &MDeviceImages.back() + 1; }

  size_t size() const { return MDeviceImages.size(); }

  bundle_state getBundleState() const {
    return MDeviceImages.empty()
               ? bundle_state::input
               : detail::getSyclObjImpl(MDeviceImages[0])->get_state();
  }

private:
  context MContext;
  std::vector<device> MDevices;
  std::vector<device_image_plain> MDeviceImages;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
