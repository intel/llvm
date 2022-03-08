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
#include <detail/mem_alloc_helper.hpp>
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

template <class T> struct LessByHash {
  bool operator()(const T &LHS, const T &RHS) const {
    return getSyclObjImpl(LHS) < getSyclObjImpl(RHS);
  }
};

// The class is impl counterpart for sycl::device_image
// It can represent a program in different states, kernel_id's it has and state
// of specialization constants for it
class device_image_impl {
public:
  // The struct maps specialization ID to offset in the binary blob where value
  // for this spec const should be.
  struct SpecConstDescT {
    unsigned int ID = 0;
    unsigned int CompositeOffset = 0;
    unsigned int Size = 0;
    unsigned int BlobOffset = 0;
    bool IsSet = false;
  };

  using SpecConstMapT = std::map<std::string, std::vector<SpecConstDescT>>;

  device_image_impl(const RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State,
                    std::shared_ptr<std::vector<kernel_id>> KernelIDs,
                    RT::PiProgram Program)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State), MProgram(Program),
        MKernelIDs(std::move(KernelIDs)) {
    updateSpecConstSymMap();
  }

  device_image_impl(const RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State,
                    std::shared_ptr<std::vector<kernel_id>> KernelIDs,
                    RT::PiProgram Program, const SpecConstMapT &SpecConstMap,
                    const std::vector<unsigned char> &SpecConstsBlob)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State), MProgram(Program),
        MKernelIDs(std::move(KernelIDs)), MSpecConstsBlob(SpecConstsBlob),
        MSpecConstSymMap(SpecConstMap) {}

  bool has_kernel(const kernel_id &KernelIDCand) const noexcept {
    return std::binary_search(MKernelIDs->begin(), MKernelIDs->end(),
                              KernelIDCand, LessByHash<kernel_id>{});
  }

  bool has_kernel(const kernel_id &KernelIDCand,
                  const device &DeviceCand) const noexcept {
    for (const device &Device : MDevices)
      if (Device == DeviceCand)
        return has_kernel(KernelIDCand);

    return false;
  }

  const std::vector<kernel_id> &get_kernel_ids() const noexcept {
    return *MKernelIDs;
  }

  bool has_specialization_constants() const noexcept {
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
    return !MSpecConstSymMap.empty();
  }

  bool all_specialization_constant_native() const noexcept {
    assert(false && "Not implemented");
    return false;
  }

  bool has_specialization_constant(const char *SpecName) const noexcept {
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
    return MSpecConstSymMap.count(SpecName) != 0;
  }

  void set_specialization_constant_raw_value(const char *SpecName,
                                             const void *Value) noexcept {
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);

    if (MSpecConstSymMap.count(std::string{SpecName}) == 0)
      return;

    std::vector<SpecConstDescT> &Descs =
        MSpecConstSymMap[std::string{SpecName}];
    for (SpecConstDescT &Desc : Descs) {
      Desc.IsSet = true;
      std::memcpy(MSpecConstsBlob.data() + Desc.BlobOffset,
                  static_cast<const char *>(Value) + Desc.CompositeOffset,
                  Desc.Size);
    }
  }

  void get_specialization_constant_raw_value(const char *SpecName,
                                             void *ValueRet) const noexcept {
    assert(is_specialization_constant_set(SpecName));
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);

    // operator[] can't be used here, since it's not marked as const
    const std::vector<SpecConstDescT> &Descs =
        MSpecConstSymMap.at(std::string{SpecName});
    for (const SpecConstDescT &Desc : Descs) {

      std::memcpy(static_cast<char *>(ValueRet) + Desc.CompositeOffset,
                  MSpecConstsBlob.data() + Desc.BlobOffset, Desc.Size);
    }
  }

  bool is_specialization_constant_set(const char *SpecName) const noexcept {
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
    if (MSpecConstSymMap.count(std::string{SpecName}) == 0)
      return false;

    const std::vector<SpecConstDescT> &Descs =
        MSpecConstSymMap.at(std::string{SpecName});
    return Descs.front().IsSet;
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

  std::shared_ptr<std::vector<kernel_id>> &get_kernel_ids_ptr() noexcept {
    return MKernelIDs;
  }

  std::vector<unsigned char> &get_spec_const_blob_ref() noexcept {
    return MSpecConstsBlob;
  }

  RT::PiMem &get_spec_const_buffer_ref() noexcept {
    std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
    if (nullptr == MSpecConstsBuffer && !MSpecConstsBlob.empty()) {
      // Uses PI_MEM_FLAGS_HOST_PTR_COPY instead of PI_MEM_FLAGS_HOST_PTR_USE
      // since post-enqueue cleanup might trigger destruction of
      // device_image_impl and, as a result, destruction of MSpecConstsBlob
      // while MSpecConstsBuffer is still in use.
      // TODO consider changing the lifetime of device_image_impl instead
      memBufferCreateHelper(detail::getSyclObjImpl(MContext),
                            PI_MEM_FLAGS_ACCESS_RW | PI_MEM_FLAGS_HOST_PTR_COPY,
                            MSpecConstsBlob.size(), MSpecConstsBlob.data(),
                            &MSpecConstsBuffer, nullptr);
    }
    return MSpecConstsBuffer;
  }

  const SpecConstMapT &get_spec_const_data_ref() const noexcept {
    return MSpecConstSymMap;
  }

  std::mutex &get_spec_const_data_lock() noexcept {
    return MSpecConstAccessMtx;
  }

  pi_native_handle getNative() const {
    assert(MProgram);
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
    if (MSpecConstsBuffer) {
      std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
      const detail::plugin &Plugin = getSyclObjImpl(MContext)->getPlugin();
      memReleaseHelper(Plugin, MSpecConstsBuffer);
    }
  }

private:
  void updateSpecConstSymMap() {
    if (MBinImage) {
      const pi::DeviceBinaryImage::PropertyRange &SCRange =
          MBinImage->getSpecConstants();
      using SCItTy = pi::DeviceBinaryImage::PropertyRange::ConstIterator;

      // get default values for specialization constants
      const pi::DeviceBinaryImage::PropertyRange &SCDefValRange =
          MBinImage->getSpecConstantsDefaultValues();

      // This variable is used to calculate spec constant value offset in a
      // flat byte array.
      unsigned BlobOffset = 0;
      for (SCItTy SCIt : SCRange) {
        const char *SCName = (*SCIt)->Name;

        pi::ByteArray Descriptors =
            pi::DeviceBinaryProperty(*SCIt).asByteArray();
        assert(Descriptors.size() > 8 && "Unexpected property size");

        // Expected layout is vector of 3-component tuples (flattened into a
        // vector of scalars), where each tuple consists of: ID of a scalar spec
        // constant, (which might be a member of the composite); offset, which
        // is used to calculate location of scalar member within the composite
        // or zero for scalar spec constants; size of a spec constant
        constexpr size_t NumElements = 3;
        assert(((Descriptors.size() - 8) / sizeof(std::uint32_t)) %
                       NumElements ==
                   0 &&
               "unexpected layout of composite spec const descriptors");
        auto *It = reinterpret_cast<const std::uint32_t *>(&Descriptors[8]);
        auto *End = reinterpret_cast<const std::uint32_t *>(&Descriptors[0] +
                                                            Descriptors.size());
        unsigned LocalOffset = 0;
        while (It != End) {
          // Make sure that alignment is correct in blob.
          const unsigned OffsetFromLast = /*Offset*/ It[1] - LocalOffset;
          BlobOffset += OffsetFromLast;
          // Composites may have a special padding element at the end which
          // should not have a descriptor. These padding elements all have max
          // ID value.
          if (It[0] != std::numeric_limits<std::uint32_t>::max()) {
            // The map is not locked here because updateSpecConstSymMap() is
            // only supposed to be called from c'tor.
            MSpecConstSymMap[std::string{SCName}].push_back(
                SpecConstDescT{/*ID*/ It[0], /*CompositeOffset*/ It[1],
                               /*Size*/ It[2], BlobOffset});
          }
          LocalOffset += OffsetFromLast + /*Size*/ It[2];
          BlobOffset += /*Size*/ It[2];
          It += NumElements;
        }
      }
      MSpecConstsBlob.resize(BlobOffset);

      bool HasDefaultValues = SCDefValRange.begin() != SCDefValRange.end();

      if (HasDefaultValues) {
        pi::ByteArray DefValDescriptors =
            pi::DeviceBinaryProperty(*SCDefValRange.begin()).asByteArray();
        assert(DefValDescriptors.size() - 8 == MSpecConstsBlob.size() &&
               "Specialization constant default value blob do not have the "
               "expected size.");
        std::uninitialized_copy(&DefValDescriptors[8],
                                &DefValDescriptors[8] + MSpecConstsBlob.size(),
                                MSpecConstsBlob.data());
      }
    }
  }

  const RTDeviceBinaryImage *MBinImage = nullptr;
  context MContext;
  std::vector<device> MDevices;
  bundle_state MState;
  // Native program handler which this device image represents
  RT::PiProgram MProgram = nullptr;
  // List of kernel ids available in this image, elements should be sorted
  // according to LessByNameComp
  std::shared_ptr<std::vector<kernel_id>> MKernelIDs;

  // A mutex for sycnhronizing access to spec constants blob. Mutable because
  // needs to be locked in the const method for getting spec constant value.
  mutable std::mutex MSpecConstAccessMtx;
  // Binary blob which can have values of all specialization constants in the
  // image
  std::vector<unsigned char> MSpecConstsBlob;
  // Buffer containing binary blob which can have values of all specialization
  // constants in the image, it is using for storing non-native specialization
  // constants
  RT::PiMem MSpecConstsBuffer = nullptr;
  // Contains map of spec const names to their descriptions + offsets in
  // the MSpecConstsBlob
  std::map<std::string, std::vector<SpecConstDescT>> MSpecConstSymMap;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
