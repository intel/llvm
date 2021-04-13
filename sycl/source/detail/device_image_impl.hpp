//==------- device_image_impl.hpp - SYCL device_image_impl -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_id_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// The class is impl counterpart for sycl::device_image
// It can represent a program in different states, kernel_id's it has and state
// of specialization constants for it
class device_image_impl {
public:
  device_image_impl(const RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State,
                    std::vector<kernel_id> KernelIDs, RT::PiProgram Program)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State), MProgram(Program),
        MKernelIDs(std::move(KernelIDs)) {}

  bool has_kernel(const kernel_id &KernelIDCand) const noexcept {
    return std::binary_search(MKernelIDs.begin(), MKernelIDs.end(),
                              KernelIDCand, LessByNameComp{});
  }

  bool has_kernel(const kernel_id &KernelIDCand,
                  const device &DeviceCand) const noexcept {
    for (const device &Device : MDevices)
      if (Device == DeviceCand)
        return has_kernel(KernelIDCand);

    return false;
  }

  const std::vector<kernel_id> &get_kernel_ids() const noexcept {
    return MKernelIDs;
  }

  bool has_specialization_constants() const noexcept {
    return !MSpecConstsBlob.empty();
  }

  bool all_specialization_constant_native() const noexcept {
    assert(false && "Not implemented");
    return false;
  }

  // The struct maps specialization ID to offset in the binary blob where value
  // for this spec const should be.
  struct SpecConstDescT {
    unsigned int ID = 0;
    unsigned int Offset = 0;
    bool IsSet = false;
  };

  bool has_specialization_constant(unsigned int SpecID) const noexcept {
    return std::any_of(MSpecConstDescs.begin(), MSpecConstDescs.end(),
                       [SpecID](const SpecConstDescT &SpecConstDesc) {
                         return SpecConstDesc.ID == SpecID;
                       });
  }

  void set_specialization_constant_raw_value(unsigned int SpecID,
                                             const void *Value,
                                             size_t ValueSize) noexcept {
    for (const SpecConstDescT &SpecConstDesc : MSpecConstDescs)
      if (SpecConstDesc.ID == SpecID) {
        // Lock the mutex to prevent when one thread in the middle of writing a
        // new value while another thread is reading the value to pass it to
        // JIT compiler.
        const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
        std::memcpy(MSpecConstsBlob.data() + SpecConstDesc.Offset, Value,
                    ValueSize);
        return;
      }
  }

  void get_specialization_constant_raw_value(unsigned int SpecID,
                                             void *ValueRet,
                                             size_t ValueSize) const noexcept {
    for (const SpecConstDescT &SpecConstDesc : MSpecConstDescs)
      if (SpecConstDesc.ID == SpecID) {
        // Lock the mutex to prevent when one thread in the middle of writing a
        // new value while another thread is reading the value to pass it to
        // JIT compiler.
        const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
        std::memcpy(ValueRet, MSpecConstsBlob.data() + SpecConstDesc.Offset,
                    ValueSize);
        return;
      }
  }

  bundle_state get_state() const noexcept { return MState; }

  void set_state(bundle_state NewState) noexcept { MState = NewState; }

  const std::vector<device> &get_devices() const noexcept { return MDevices; }

  bool compatible_with_device(const device &Dev) const {
    return std::any_of(
        MDevices.begin(), MDevices.end(),
        [&Dev](const device &DevCand) { return Dev == DevCand; });
  }

  const RT::PiProgram &get_program_ref() const noexcept { return MProgram; }

  const RTDeviceBinaryImage *&get_bin_image_ref() noexcept { return MBinImage; }

  const context &get_context() const noexcept { return MContext; }

  std::vector<kernel_id> &get_kernel_ids_ref() noexcept { return MKernelIDs; }

  std::vector<unsigned char> &get_spec_const_blob_ref() noexcept {
    return MSpecConstsBlob;
  }

  std::vector<SpecConstDescT> &get_spec_const_offsets_ref() noexcept {
    return MSpecConstDescs;
  }

  pi_native_handle getNative() const {
    const auto &ContextImplPtr = detail::getSyclObjImpl(MContext);
    const plugin &Plugin = ContextImplPtr->getPlugin();

    pi_native_handle NativeProgram = 0;
    Plugin.call<PiApiKind::piextProgramGetNativeHandle>(MProgram,
                                                        &NativeProgram);

    return NativeProgram;
  }

  ~device_image_impl() {

    if (MProgram) {
      const detail::plugin &Plugin = getSyclObjImpl(MContext)->getPlugin();
      Plugin.call<PiApiKind::piProgramRelease>(MProgram);
    }
  }

private:
  const RTDeviceBinaryImage *MBinImage = nullptr;
  context MContext;
  std::vector<device> MDevices;
  bundle_state MState;
  // Native program handler which this device image represents
  RT::PiProgram MProgram = nullptr;
  // List of kernel ids available in this image, elements should be sorted
  // according to LessByNameComp
  std::vector<kernel_id> MKernelIDs;

  // A mutex for sycnhronizing access to spec constants blob. Mutable because
  // needs to be locked in the const method for getting spec constant value.
  mutable std::mutex MSpecConstAccessMtx;
  // Binary blob which can have values of all specialization constants in the
  // image
  std::vector<unsigned char> MSpecConstsBlob;
  // Contains list of spec ID + their offsets in the MSpecConstsBlob
  std::vector<SpecConstDescT> MSpecConstDescs;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
