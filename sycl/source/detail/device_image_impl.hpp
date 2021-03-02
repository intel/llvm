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
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class device_image_impl {
public:
  device_image_impl(RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State) {

    // Collect kernel names for the image
    pi_device_binary DevBin =
        const_cast<pi_device_binary>(&BinImage->getRawData());
    for (_pi_offload_entry EntriesIt = DevBin->EntriesBegin;
         EntriesIt != DevBin->EntriesEnd; ++EntriesIt) {

      std::shared_ptr<detail::kernel_id_impl> KernleIDImpl =
          std::make_shared<detail::kernel_id_impl>(EntriesIt->name);
      MKernelIDs.emplace_back(
          detail::createSyclObjFromImpl<sycl::kernel_id>(KernleIDImpl));
    }
  }

  bool has_kernel(const kernel_id &KernelIDCand) const noexcept {
    return std::any_of(MKernelIDs.begin(), MKernelIDs.end(),
                       [&KernelIDCand](const kernel_id &KernelID) {
                         return strcmp(KernelID.get_name(),
                                       KernelIDCand.get_name()) == 0;
                       });
  }

  bool has_kernel(const kernel_id &KernelIDCand,
                  const device &DeviceCand) const noexcept {
    for (const device &Device : MDevices)
      if (Device == DeviceCand)
        return has_kernel(KernelIDCand);

    return false;
  }

  const std::vector<kernel_id> &getKernelIDs() const { return MKernelIDs; }

  bool hasSpecConsts() const { return !MSpecConstsBlob.empty(); }

  bool allSpecConstNative() const {
    assert(false && "Not implemented");
    return false;
  }

  // The struct maps specialization ID to offset in the binary blob where value
  // for this spec const should be.
  struct SpecConstIDOffset {
    unsigned int ID = 0;
    unsigned int Offset = 0;
  };

  bool hasSpecConst(unsigned int SpecID) const noexcept {
    return std::any_of(
        MSpecConstOffsets.begin(), MSpecConstOffsets.end(),
        [SpecID](const SpecConstIDOffset &Pair) { return Pair.ID == SpecID; });
  }

  void set_specialization_constant(unsigned int SpecID, const void *Value,
                                   size_t ValueSize) {
    for (const SpecConstIDOffset &Pair : MSpecConstOffsets)
      if (Pair.ID == SpecID) {
        memcpy(MSpecConstsBlob.data() + Pair.Offset, Value, ValueSize);
        return;
      }
  }

  const void *get_specialization_constant(unsigned int SpecID) const {
    for (const SpecConstIDOffset &Pair : MSpecConstOffsets)
      if (Pair.ID == SpecID)
        return MSpecConstsBlob.data() + Pair.Offset;

    return nullptr;
  }

  bundle_state getState() const { return MState; }

  void setState(bundle_state NewState) { MState = NewState; }

private:
  RTDeviceBinaryImage *MBinImage = nullptr;
  context MContext;
  std::vector<device> MDevices;
  bundle_state MState;
  // List of kernel ids available in this image
  std::vector<kernel_id> MKernelIDs;

  // Binary blob which can have values of all specialization constants in the
  // image
  std::vector<unsigned char> MSpecConstsBlob;
  // Contains list of spec ID + their offsets in the MSpecConstsBlob
  std::vector<SpecConstIDOffset> MSpecConstOffsets;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
