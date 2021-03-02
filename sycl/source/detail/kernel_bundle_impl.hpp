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

class kernel_bundle_impl {

public:
  kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                     bundle_state State)
      : MContext(Ctx), MDevices(Devs) {

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        Ctx, Devs, State);
  }

  kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                     const std::vector<kernel_id> &KernelIDs,
                     bundle_state State)
      : kernel_bundle_impl(Ctx, Devs, State) {

    // Filter out images that have no kernel_ids specified
    auto It = std::remove_if(MDeviceImages.begin(), MDeviceImages.end(),
                             [&KernelIDs](const device_image_plain &Image) {
                               const auto It = std::find_if(
                                   KernelIDs.begin(), KernelIDs.end(),
                                   [&Image](const sycl::kernel_id &KernelID) {
                                     return Image.has_kernel(KernelID);
                                   });
                               const bool ContainsKernels =
                                   (It != KernelIDs.end());
                               return !ContainsKernels;
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

  bool empty() const noexcept { return MDeviceImages.empty(); }

  backend get_backend() const noexcept {
    return MContext.get_platform().get_backend();
  }

  context get_context() const noexcept { return MContext; }

  std::vector<device> get_devices() const noexcept { return MDevices; }

  std::vector<kernel_id> get_kernel_ids() const {
    // Collect kernel ids from all device images, then remove duplicates

    std::vector<kernel_id> Result;
    for (const device_image_plain &DeviceImage : MDeviceImages) {
      const std::vector<kernel_id> &KernelIDs =
          getSyclObjImpl(DeviceImage)->getKernelIDs();

      Result.insert(Result.end(), KernelIDs.begin(), KernelIDs.end());
    }
    std::sort(Result.begin(), Result.end(),
              [](const kernel_id &LHS, const kernel_id &RHS) {
                return detail::getSyclObjImpl(LHS) <
                       detail::getSyclObjImpl(RHS);
              });
    std::unique(Result.begin(), Result.end());

    return Result;
  }

  kernel get_kernel(const kernel_id &KernelID) const {
    assert(false && "Not implemented");
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
    return std::any_of(MDeviceImages.begin(), MDeviceImages.end(),
                       [](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)->hasSpecConsts();
                       });
  }

  bool native_specialization_constant() const noexcept {
    return std::all_of(
        MDeviceImages.begin(), MDeviceImages.end(),
        [](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->allSpecConstNative();
        });
  }

  bool has_specialization_constant(unsigned int SpecID) const noexcept {
    return std::any_of(
        MDeviceImages.begin(), MDeviceImages.end(),
        [SpecID](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->hasSpecConst(SpecID);
        });
  }

  void set_specialization_constant(unsigned int SpecID, const void *Value,
                                   size_t ValueSize) {
    for (const device_image_plain &DeviceImage : MDeviceImages)
      getSyclObjImpl(DeviceImage)
          ->set_specialization_constant(SpecID, Value, ValueSize);
  }

  const void *get_specialization_constant(unsigned int SpecID) const {
    for (const device_image_plain &DeviceImage : MDeviceImages)
      if (const void *Value =
              getSyclObjImpl(DeviceImage)->get_specialization_constant(SpecID))
        return Value;

    return nullptr;
  }

  const device_image_plain *begin() const { return &MDeviceImages.front(); }

  const device_image_plain *end() const { return &MDeviceImages.back() + 1; }

private:
  context MContext;
  std::vector<device> MDevices;
  std::vector<device_image_plain> MDeviceImages;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
