//==------- device_image_impl.hpp - SYCL device_image_impl -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_id_impl.hpp>
#include <detail/mem_alloc_helper.hpp>
#include <detail/plugin.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace sycl {
inline namespace _V1 {
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
    // Indicates if the specialization constant was set to a value which is
    // different from the default value.
    bool IsSet = false;
  };

  using SpecConstMapT = std::map<std::string, std::vector<SpecConstDescT>>;

  device_image_impl(const RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State,
                    std::shared_ptr<std::vector<kernel_id>> KernelIDs,
                    sycl::detail::pi::PiProgram Program)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State), MProgram(Program),
        MKernelIDs(std::move(KernelIDs)),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()) {
    updateSpecConstSymMap();
  }

  device_image_impl(const RTDeviceBinaryImage *BinImage, context Context,
                    std::vector<device> Devices, bundle_state State,
                    std::shared_ptr<std::vector<kernel_id>> KernelIDs,
                    sycl::detail::pi::PiProgram Program,
                    const SpecConstMapT &SpecConstMap,
                    const std::vector<unsigned char> &SpecConstsBlob)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(std::move(Devices)), MState(State), MProgram(Program),
        MKernelIDs(std::move(KernelIDs)), MSpecConstsBlob(SpecConstsBlob),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MSpecConstSymMap(SpecConstMap) {}

  bool has_kernel(const kernel_id &KernelIDCand) const noexcept {
    return std::binary_search(MKernelIDs->begin(), MKernelIDs->end(),
                              KernelIDCand, LessByHash<kernel_id>{});
  }

  bool has_kernel(const kernel_id &KernelIDCand,
                  const device &DeviceCand) const noexcept {
    // If the device is in the device list and the kernel ID is in the kernel
    // bundle, return true.
    for (const device &Device : MDevices)
      if (Device == DeviceCand)
        return has_kernel(KernelIDCand);

    // Otherwise, if the device candidate is a sub-device it is also valid if
    // its parent is valid.
    if (!getSyclObjImpl(DeviceCand)->isRootDevice())
      return has_kernel(KernelIDCand,
                        DeviceCand.get_info<info::device::parent_device>());

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
    // Specialization constants are natively supported in JIT mode on backends,
    // that are using SPIR-V as IR

    // Not sure if it's possible currently, but probably it may happen if the
    // kernel bundle is created with interop function. Now the only one such
    // function is make_kernel(), but I'm not sure if it's even possible to
    // use spec constant with such kernel. So, in such case we need to check
    // if it's JIT or no somehow.
    assert(MBinImage &&
           "native_specialization_constant() called for unimplemented case");

    auto IsJITSPIRVTarget = [](const char *Target) {
      return (strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0 ||
              strcmp(Target, __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV32) == 0);
    };
    return (MContext.get_backend() == backend::opencl ||
            MContext.get_backend() == backend::ext_oneapi_level_zero) &&
           IsJITSPIRVTarget(MBinImage->getRawData().DeviceTargetSpec);
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
      // If there is a default value of the specialization constant and it is
      // the same as the value which is being set then do nothing, runtime is
      // going to handle this case just like if only the default value of the
      // specialization constant was provided.
      if (MSpecConstsDefValBlob.size() &&
          (std::memcmp(MSpecConstsDefValBlob.begin() + Desc.BlobOffset,
                       static_cast<const char *>(Value) + Desc.CompositeOffset,
                       Desc.Size) == 0)) {
        // Now we have default value, so reset to false.
        Desc.IsSet = false;
        continue;
      }

      // Value of the specialization constant is set to a value which is
      // different from the default value.
      Desc.IsSet = true;
      std::memcpy(MSpecConstsBlob.data() + Desc.BlobOffset,
                  static_cast<const char *>(Value) + Desc.CompositeOffset,
                  Desc.Size);
    }
  }

  void get_specialization_constant_raw_value(const char *SpecName,
                                             void *ValueRet) const noexcept {
    bool IsSet = is_specialization_constant_set(SpecName);
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
    assert(IsSet || MSpecConstsDefValBlob.size());
    // operator[] can't be used here, since it's not marked as const
    const std::vector<SpecConstDescT> &Descs =
        MSpecConstSymMap.at(std::string{SpecName});
    for (const SpecConstDescT &Desc : Descs) {
      auto Blob =
          IsSet ? MSpecConstsBlob.data() : MSpecConstsDefValBlob.begin();
      std::memcpy(static_cast<char *>(ValueRet) + Desc.CompositeOffset,
                  Blob + Desc.BlobOffset, Desc.Size);
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

  bool is_any_specialization_constant_set() const noexcept {
    // Lock the mutex to prevent when one thread in the middle of writing a
    // new value while another thread is reading the value to pass it to
    // JIT compiler.
    const std::lock_guard<std::mutex> SpecConstLock(MSpecConstAccessMtx);
    for (auto &SpecConst : MSpecConstSymMap) {
      for (auto &Desc : SpecConst.second) {
        if (Desc.IsSet)
          return true;
      }
    }

    return false;
  }

  bool specialization_constants_replaced_with_default() const noexcept {
    pi_device_binary_property Prop =
        MBinImage->getProperty("specConstsReplacedWithDefault");
    return Prop && (DeviceBinaryProperty(Prop).asUint32() != 0);
  }

  bundle_state get_state() const noexcept { return MState; }

  void set_state(bundle_state NewState) noexcept { MState = NewState; }

  const std::vector<device> &get_devices() const noexcept { return MDevices; }

  bool compatible_with_device(const device &Dev) const {
    return std::any_of(
        MDevices.begin(), MDevices.end(),
        [&Dev](const device &DevCand) { return Dev == DevCand; });
  }

  const sycl::detail::pi::PiProgram &get_program_ref() const noexcept {
    return MProgram;
  }

  const RTDeviceBinaryImage *&get_bin_image_ref() noexcept { return MBinImage; }

  const context &get_context() const noexcept { return MContext; }

  std::shared_ptr<std::vector<kernel_id>> &get_kernel_ids_ptr() noexcept {
    return MKernelIDs;
  }

  std::vector<unsigned char> &get_spec_const_blob_ref() noexcept {
    return MSpecConstsBlob;
  }

  sycl::detail::pi::PiMem &get_spec_const_buffer_ref() noexcept {
    std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
    if (nullptr == MSpecConstsBuffer && !MSpecConstsBlob.empty()) {
      const PluginPtr &Plugin = getSyclObjImpl(MContext)->getPlugin();
      // Uses PI_MEM_FLAGS_HOST_PTR_COPY instead of PI_MEM_FLAGS_HOST_PTR_USE
      // since post-enqueue cleanup might trigger destruction of
      // device_image_impl and, as a result, destruction of MSpecConstsBlob
      // while MSpecConstsBuffer is still in use.
      // TODO consider changing the lifetime of device_image_impl instead
      memBufferCreateHelper(Plugin,
                            detail::getSyclObjImpl(MContext)->getHandleRef(),
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
    const PluginPtr &Plugin = ContextImplPtr->getPlugin();

    if (ContextImplPtr->getBackend() == backend::opencl)
      Plugin->call<PiApiKind::piProgramRetain>(MProgram);
    pi_native_handle NativeProgram = 0;
    Plugin->call<PiApiKind::piextProgramGetNativeHandle>(MProgram,
                                                         &NativeProgram);

    return NativeProgram;
  }

  ~device_image_impl() {
    try {
      if (MProgram) {
        const PluginPtr &Plugin = getSyclObjImpl(MContext)->getPlugin();
        Plugin->call<PiApiKind::piProgramRelease>(MProgram);
      }
      if (MSpecConstsBuffer) {
        std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
        const PluginPtr &Plugin = getSyclObjImpl(MContext)->getPlugin();
        memReleaseHelper(Plugin, MSpecConstsBuffer);
      }
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~device_image_impl", e);
    }
  }

private:
  // Get the specialization constant default value blob.
  ByteArray getSpecConstsDefValBlob() const {
    if (!MBinImage)
      return ByteArray(nullptr, 0);

    // Get default values for specialization constants.
    const RTDeviceBinaryImage::PropertyRange &SCDefValRange =
        MBinImage->getSpecConstantsDefaultValues();
    if (!SCDefValRange.size())
      return ByteArray(nullptr, 0);

    ByteArray DefValDescriptors =
        DeviceBinaryProperty(*SCDefValRange.begin()).asByteArray();
    // First 8 bytes are consumed by the size of the property.
    DefValDescriptors.dropBytes(8);
    return DefValDescriptors;
  }

  void updateSpecConstSymMap() {
    if (MBinImage) {
      const RTDeviceBinaryImage::PropertyRange &SCRange =
          MBinImage->getSpecConstants();
      using SCItTy = RTDeviceBinaryImage::PropertyRange::ConstIterator;

      // This variable is used to calculate spec constant value offset in a
      // flat byte array.
      unsigned BlobOffset = 0;
      for (SCItTy SCIt : SCRange) {
        const char *SCName = (*SCIt)->Name;

        ByteArray Descriptors = DeviceBinaryProperty(*SCIt).asByteArray();
        // First 8 bytes are consumed by the size of the property.
        Descriptors.dropBytes(8);

        // Expected layout is vector of 3-component tuples (flattened into a
        // vector of scalars), where each tuple consists of: ID of a scalar spec
        // constant, (which might be a member of the composite); offset, which
        // is used to calculate location of scalar member within the composite
        // or zero for scalar spec constants; size of a spec constant.
        unsigned LocalOffset = 0;
        while (!Descriptors.empty()) {
          auto [Id, CompositeOffset, Size] =
              Descriptors.consume<uint32_t, uint32_t, uint32_t>();

          // Make sure that alignment is correct in the blob.
          const unsigned OffsetFromLast = CompositeOffset - LocalOffset;
          BlobOffset += OffsetFromLast;
          // Composites may have a special padding element at the end which
          // should not have a descriptor. These padding elements all have max
          // ID value.
          if (Id != std::numeric_limits<std::uint32_t>::max()) {
            // The map is not locked here because updateSpecConstSymMap() is
            // only supposed to be called from c'tor.
            MSpecConstSymMap[std::string{SCName}].push_back(
                SpecConstDescT{Id, CompositeOffset, Size, BlobOffset});
          }
          LocalOffset += OffsetFromLast + Size;
          BlobOffset += Size;
        }
      }
      MSpecConstsBlob.resize(BlobOffset);

      if (MSpecConstsDefValBlob.size()) {
        assert(MSpecConstsDefValBlob.size() == MSpecConstsBlob.size() &&
               "Specialization constant default value blob do not have the "
               "expected size.");
        std::uninitialized_copy(MSpecConstsDefValBlob.begin(),
                                MSpecConstsDefValBlob.begin() +
                                    MSpecConstsBlob.size(),
                                MSpecConstsBlob.data());
      }
    }
  }

  const RTDeviceBinaryImage *MBinImage = nullptr;
  context MContext;
  std::vector<device> MDevices;
  bundle_state MState;
  // Native program handler which this device image represents
  sycl::detail::pi::PiProgram MProgram = nullptr;
  // List of kernel ids available in this image, elements should be sorted
  // according to LessByNameComp
  std::shared_ptr<std::vector<kernel_id>> MKernelIDs;

  // A mutex for sycnhronizing access to spec constants blob. Mutable because
  // needs to be locked in the const method for getting spec constant value.
  mutable std::mutex MSpecConstAccessMtx;
  // Binary blob which can have values of all specialization constants in the
  // image
  std::vector<unsigned char> MSpecConstsBlob;
  // Binary blob which can have default values of all specialization constants
  // in the image.
  const ByteArray MSpecConstsDefValBlob;
  // Buffer containing binary blob which can have values of all specialization
  // constants in the image, it is using for storing non-native specialization
  // constants
  sycl::detail::pi::PiMem MSpecConstsBuffer = nullptr;
  // Contains map of spec const names to their descriptions + offsets in
  // the MSpecConstsBlob
  std::map<std::string, std::vector<SpecConstDescT>> MSpecConstSymMap;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
