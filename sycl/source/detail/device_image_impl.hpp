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
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_id_impl.hpp>
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

// Used for sorting vector of kernel_id's
struct LessByNameComp {
  bool operator()(const sycl::kernel_id &LHS, const sycl::kernel_id &RHS) {
    return std::strcmp(LHS.get_name(), RHS.get_name()) < 0;
  }
};

// The class is impl counterpart for sycl::device_image
// It can represent a program in different states, kernel_id's it has and state
// of specialization constants for it
class device_image_impl {
public:
  device_image_impl(RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State) {

    // Collect kernel names for the image
    pi_device_binary DevBin =
        const_cast<pi_device_binary>(&MBinImage->getRawData());
    for (_pi_offload_entry EntriesIt = DevBin->EntriesBegin;
         EntriesIt != DevBin->EntriesEnd; ++EntriesIt) {

      std::shared_ptr<detail::kernel_id_impl> KernleIDImpl =
          std::make_shared<detail::kernel_id_impl>(EntriesIt->name);

      sycl::kernel_id KernelID =
          detail::createSyclObjFromImpl<sycl::kernel_id>(KernleIDImpl);

      // Insert new element keeping MKernelIDs sorted.
      auto It = std::lower_bound(MKernelIDs.begin(), MKernelIDs.end(), KernelID,
                                 LessByNameComp{});
      MKernelIDs.insert(It, std::move(KernelID));
    }
  }

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
  struct SpecConstIDOffset {
    unsigned int ID = 0;
    unsigned int Offset = 0;
  };

  bool has_specialization_constant(unsigned int SpecID) const noexcept {
    return std::any_of(
        MSpecConstOffsets.begin(), MSpecConstOffsets.end(),
        [SpecID](const SpecConstIDOffset &Pair) { return Pair.ID == SpecID; });
  }

  void set_specialization_constant_raw_value(unsigned int SpecID,
                                             const void *Value,
                                             size_t ValueSize) noexcept {
    for (const SpecConstIDOffset &Pair : MSpecConstOffsets)
      if (Pair.ID == SpecID) {
        // Lock the mutex to prevent when one thread in the middle of writing a
        // new value while another thread is reading the value to pass it to
        // JIT compiler.
        const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
        std::memcpy(MSpecConstsBlob.data() + Pair.Offset, Value, ValueSize);
        return;
      }
  }

  void get_specialization_constant_raw_value(unsigned int SpecID,
                                             void *ValueRet,
                                             size_t ValueSize) const noexcept {
    for (const SpecConstIDOffset &Pair : MSpecConstOffsets)
      if (Pair.ID == SpecID) {
        // Lock the mutex to prevent when one thread in the middle of writing a
        // new value while another thread is reading the value to pass it to
        // JIT compiler.
        const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
        std::memcpy(ValueRet, MSpecConstsBlob.data() + Pair.Offset, ValueSize);
        return;
      }
  }

  bundle_state get_state() const noexcept { return MState; }

  void set_state(bundle_state NewState) noexcept { MState = NewState; }

private:
  RTDeviceBinaryImage *MBinImage = nullptr;
  context MContext;
  std::vector<device> MDevices;
  bundle_state MState;
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
  std::vector<SpecConstIDOffset> MSpecConstOffsets;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
