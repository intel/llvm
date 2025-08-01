//==------- device_image_impl.hpp - SYCL device_image_impl -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if SYCL_EXT_JIT_ENABLE
#include "JITBinaryInfo.h"
#endif // SYCL_EXT_JIT_ENABLE
#include <detail/adapter_impl.hpp>
#include <detail/compiler.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_compiler/kernel_compiler_opencl.hpp>
#include <detail/kernel_compiler/kernel_compiler_sycl.hpp>
#include <detail/kernel_id_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/mem_alloc_helper.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/split_string.hpp>
#include <detail/syclbin.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace jit_compiler {
enum class BinaryFormat : uint32_t;
}

namespace sycl {
inline namespace _V1 {
namespace detail {

template <class T> struct LessByHash {
  bool operator()(const T &LHS, const T &RHS) const {
    return getSyclObjImpl(LHS) < getSyclObjImpl(RHS);
  }
};

namespace syclex = sycl::ext::oneapi::experimental;

using include_pairs_t =
    std::vector<std::pair<std::string /* name */, std::string /* content */>>;

// Bits representing the origin of a given image, i.e. regular offline SYCL
// compilation, interop, kernel_compiler online compilation, etc.
constexpr uint8_t ImageOriginSYCLOffline = 1;
constexpr uint8_t ImageOriginInterop = 1 << 1;
constexpr uint8_t ImageOriginKernelCompiler = 1 << 2;
constexpr uint8_t ImageOriginSYCLBIN = 1 << 3;

// Helper class to track and unregister shared SYCL device_globals.
class ManagedDeviceGlobalsRegistry {
public:
  ManagedDeviceGlobalsRegistry(
      context_impl &ContextImpl, const std::string &Prefix,
      std::vector<std::string> &&DeviceGlobalNames,
      std::vector<std::unique_ptr<std::byte[]>> &&DeviceGlobalAllocations)
      : MContextImpl{ContextImpl.shared_from_this()}, MPrefix{Prefix},
        MDeviceGlobalNames{std::move(DeviceGlobalNames)},
        MDeviceGlobalAllocations{std::move(DeviceGlobalAllocations)} {}

  ManagedDeviceGlobalsRegistry(const ManagedDeviceGlobalsRegistry &) = delete;

  ManagedDeviceGlobalsRegistry &
  operator=(const ManagedDeviceGlobalsRegistry &) = delete;

  ~ManagedDeviceGlobalsRegistry() {
    try {
      unregisterDeviceGlobalsFromContext();
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM(
          "exception during unregistration of SYCL binaries", e);
    }
  }

  bool hasDeviceGlobalName(const std::string &Name) const noexcept {
    return !MDeviceGlobalNames.empty() &&
           std::find(MDeviceGlobalNames.begin(), MDeviceGlobalNames.end(),
                     Name) != MDeviceGlobalNames.end();
  }

  DeviceGlobalMapEntry *tryGetDeviceGlobalEntry(const std::string &Name) const {
    auto &PM = detail::ProgramManager::getInstance();
    return PM.tryGetDeviceGlobalEntry(MPrefix + Name);
  }

private:
  void unregisterDeviceGlobalsFromContext() {
    if (MDeviceGlobalNames.empty())
      return;

    // Manually trigger the release of resources for all device global map
    // entries associated with this runtime-compiled bundle. Normally, this
    // would happen in `~context_impl()`, however in the RTC setting, the
    // context outlives the DG map entries owned by the program manager.

    std::vector<std::string> DeviceGlobalIDs;
    std::transform(MDeviceGlobalNames.begin(), MDeviceGlobalNames.end(),
                   std::back_inserter(DeviceGlobalIDs),
                   [&](const std::string &DGName) { return MPrefix + DGName; });
    for (DeviceGlobalMapEntry *Entry :
         ProgramManager::getInstance().getDeviceGlobalEntries(
             DeviceGlobalIDs)) {
      Entry->removeAssociatedResources(MContextImpl.get());
      MContextImpl->removeAssociatedDeviceGlobal(Entry->MDeviceGlobalPtr);
    }
  }

  std::shared_ptr<context_impl> MContextImpl;

  std::string MPrefix;
  std::vector<std::string> MDeviceGlobalNames;
  std::vector<std::unique_ptr<std::byte[]>> MDeviceGlobalAllocations;
};

// Helper class to unregister shared SYCL binaries.
class ManagedDeviceBinaries {
public:
  ManagedDeviceBinaries(sycl_device_binaries &&Binaries)
      : MBinaries{Binaries} {}

  ManagedDeviceBinaries(const ManagedDeviceBinaries &) = delete;

  ManagedDeviceBinaries &operator=(const ManagedDeviceBinaries &) = delete;

  ~ManagedDeviceBinaries() {
    try {
      ProgramManager::getInstance().removeImages(MBinaries);
      syclex::detail::SYCL_JIT_Destroy(MBinaries);
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM(
          "exception during unregistration of SYCL binaries", e);
    }
  }

private:
  sycl_device_binaries MBinaries;
};

// Using ordered containers for heterogenous lookup.
// TODO change to unordered containers after switching to C++20.
using MangledKernelNameMapT = std::map<std::string, std::string, std::less<>>;
using KernelNameSetT = std::set<std::string, std::less<>>;
using KernelNameToArgMaskMap =
    std::map<std::string, KernelArgMask, std::less<>>;

// Information unique to images compiled at runtime through the
// ext_oneapi_kernel_compiler extension.
struct KernelCompilerBinaryInfo {
  KernelCompilerBinaryInfo(syclex::source_language Lang) : MLanguage{Lang} {}

  KernelCompilerBinaryInfo(syclex::source_language Lang,
                           include_pairs_t &&IncludePairsVec)
      : MLanguage{Lang}, MIncludePairs{std::move(IncludePairsVec)} {}

  KernelCompilerBinaryInfo(
      syclex::source_language Lang, MangledKernelNameMapT &&MangledKernelNames,
      std::string &&Prefix,
      std::shared_ptr<ManagedDeviceGlobalsRegistry> &&DeviceGlobalRegistry)
      : MLanguage{Lang}, MMangledKernelNames{std::move(MangledKernelNames)},
        MPrefixes{std::move(Prefix)},
        MDeviceGlobalRegistries{std::move(DeviceGlobalRegistry)} {}

  static std::optional<KernelCompilerBinaryInfo>
  Merge(const std::vector<const std::optional<KernelCompilerBinaryInfo> *>
            &RTCInfos) {
    std::optional<KernelCompilerBinaryInfo> Result = std::nullopt;
    std::map<std::string, std::string> IncludePairMap;
    for (const std::optional<KernelCompilerBinaryInfo> *RTCInfoPtr : RTCInfos) {
      if (!RTCInfoPtr || !(*RTCInfoPtr))
        continue;
      const std::optional<KernelCompilerBinaryInfo> &RTCInfo = *RTCInfoPtr;

      if (!Result) {
        Result = RTCInfo;
        continue;
      }

      if (RTCInfo->MLanguage != Result->MLanguage)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Linking binaries with different source "
                              "languages is not currently supported.");

      Result->MMangledKernelNames.insert(RTCInfo->MMangledKernelNames.begin(),
                                         RTCInfo->MMangledKernelNames.end());

      // Assumption is that there are no duplicates, but in the case we let
      // duplicates through it should be alright to pay for the minimal extra
      // space allocated.
      Result->MIncludePairs.reserve(RTCInfo->MIncludePairs.size());
      for (const auto &IncludePair : RTCInfo->MIncludePairs) {
        auto Inserted = IncludePairMap.insert(IncludePair);
        if (!Inserted.second) {
          if (Inserted.first->second != IncludePair.second)
            throw sycl::exception(make_error_code(errc::invalid),
                                  "Conflicting include files.");
        } else {
          Result->MIncludePairs.push_back(IncludePair);
        }
      }

      Result->MDeviceGlobalRegistries.insert(
          Result->MDeviceGlobalRegistries.end(),
          RTCInfo->MDeviceGlobalRegistries.begin(),
          RTCInfo->MDeviceGlobalRegistries.end());

      for (const std::string &Prefix : RTCInfo->MPrefixes)
        Result->MPrefixes.insert(Prefix);
    }
    return Result;
  }

  syclex::source_language MLanguage;
  MangledKernelNameMapT MMangledKernelNames;
  std::set<std::string> MPrefixes;
  include_pairs_t MIncludePairs;
  std::vector<std::shared_ptr<ManagedDeviceGlobalsRegistry>>
      MDeviceGlobalRegistries;
};

// The class is impl counterpart for sycl::device_image
// It can represent a program in different states, kernel_id's it has and state
// of specialization constants for it
class device_image_impl
    : public std::enable_shared_from_this<device_image_impl> {
  struct private_tag {
    explicit private_tag() = default;
  };

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
                    devices_range Devices, bundle_state State,
                    std::shared_ptr<std::vector<kernel_id>> KernelIDs,
                    Managed<ur_program_handle_t> &&Program, uint8_t Origins,
                    private_tag)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()), MState(State),
        MProgram(std::move(Program)), MKernelIDs(std::move(KernelIDs)),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()), MOrigins(Origins) {
    updateSpecConstSymMap();
    if (BinImage && (MOrigins & ImageOriginSYCLBIN)) {
      // SYCLBIN files have the kernel names embedded in the binaries, so we
      // collect them.
      for (const sycl_device_binary_property &KNProp :
           BinImage->getKernelNames())
        MKernelNames.insert(KNProp->Name);

      KernelArgMask ArgMask;
      if (BinImage->getKernelParamOptInfo().isAvailable()) {
        // Extract argument mask from the image.
        const RTDeviceBinaryImage::PropertyRange &KPOIRange =
            BinImage->getKernelParamOptInfo();
        for (const auto &Info : KPOIRange)
          MEliminatedKernelArgMasks[Info->Name] =
              createKernelArgMask(DeviceBinaryProperty(Info).asByteArray());
      }
    }
  }

  device_image_impl(
      const RTDeviceBinaryImage *BinImage, const context &Context,
      devices_range Devices, bundle_state State,
      std::shared_ptr<std::vector<kernel_id>> KernelIDs,
      Managed<ur_program_handle_t> &&Program, const SpecConstMapT &SpecConstMap,
      const std::vector<unsigned char> &SpecConstsBlob, uint8_t Origins,
      std::optional<KernelCompilerBinaryInfo> &&RTCInfo,
      KernelNameSetT &&KernelNames,
      KernelNameToArgMaskMap &&EliminatedKernelArgMasks,
      std::unique_ptr<DynRTDeviceBinaryImage> &&MergedImageStorage, private_tag)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()), MState(State),
        MProgram(std::move(Program)), MKernelIDs(std::move(KernelIDs)),
        MKernelNames{std::move(KernelNames)},
        MEliminatedKernelArgMasks{std::move(EliminatedKernelArgMasks)},
        MSpecConstsBlob(SpecConstsBlob),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MSpecConstSymMap(SpecConstMap), MOrigins(Origins),
        MRTCBinInfo(std::move(RTCInfo)),
        MMergedImageStorage(std::move(MergedImageStorage)) {}

  device_image_impl(
      const RTDeviceBinaryImage *BinImage, const context &Context,
      devices_range Devices, bundle_state State,
      std::shared_ptr<std::vector<kernel_id>> &&KernelIDs,
      syclex::source_language Lang, KernelNameSetT &&KernelNames,
      MangledKernelNameMapT &&MangledKernelNames, std::string &&Prefix,
      std::shared_ptr<ManagedDeviceGlobalsRegistry> &&DeviceGlobalRegistry,
      private_tag)
      : MBinImage(BinImage), MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()), MState(State),
        MKernelIDs(std::move(KernelIDs)), MKernelNames{std::move(KernelNames)},
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MOrigins(ImageOriginKernelCompiler),
        MRTCBinInfo(KernelCompilerBinaryInfo{
            Lang, std::move(MangledKernelNames), std::move(Prefix),
            std::move(DeviceGlobalRegistry)}) {
    updateSpecConstSymMap();
  }

  device_image_impl(const std::string &Src, context Context,
                    devices_range Devices, syclex::source_language Lang,
                    include_pairs_t &&IncludePairsVec, private_tag)
      : MBinImage(Src), MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()),
        MState(bundle_state::ext_oneapi_source),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MOrigins(ImageOriginKernelCompiler),
        MRTCBinInfo(
            KernelCompilerBinaryInfo{Lang, std::move(IncludePairsVec)}) {
    updateSpecConstSymMap();
  }

  device_image_impl(const std::vector<std::byte> &Bytes, const context &Context,
                    devices_range Devices, syclex::source_language Lang,
                    private_tag)
      : MBinImage(Bytes), MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()),
        MState(bundle_state::ext_oneapi_source),
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MOrigins(ImageOriginKernelCompiler),
        MRTCBinInfo(KernelCompilerBinaryInfo{Lang}) {
    updateSpecConstSymMap();
  }

  device_image_impl(const context &Context, devices_range Devices,
                    bundle_state State, Managed<ur_program_handle_t> &&Program,
                    syclex::source_language Lang, KernelNameSetT &&KernelNames,
                    private_tag)
      : MBinImage(static_cast<const RTDeviceBinaryImage *>(nullptr)),
        MContext(std::move(Context)),
        MDevices(Devices.to<std::vector<device_impl *>>()), MState(State),
        MProgram(std::move(Program)), MKernelNames{std::move(KernelNames)},
        MSpecConstsDefValBlob(getSpecConstsDefValBlob()),
        MOrigins(ImageOriginKernelCompiler),
        MRTCBinInfo(KernelCompilerBinaryInfo{Lang}) {}

  template <typename... Ts>
  static std::shared_ptr<device_image_impl> create(Ts &&...args) {
    return std::make_shared<device_image_impl>(std::forward<Ts>(args)...,
                                               private_tag{});
  }

  bool has_kernel(const kernel_id &KernelIDCand) const noexcept {
    if (!MKernelIDs)
      return false;
    return std::binary_search(MKernelIDs->begin(), MKernelIDs->end(),
                              KernelIDCand, LessByHash<kernel_id>{});
  }

  bool has_kernel(const kernel_id &KernelIDCand,
                  const device &DeviceCand) const noexcept {
    // If the device is in the device list and the kernel ID is in the kernel
    // bundle, return true.
    if (get_devices().contains(*getSyclObjImpl(DeviceCand)))
      return has_kernel(KernelIDCand);

    // Otherwise, if the device candidate is a sub-device it is also valid if
    // its parent is valid.
    if (!getSyclObjImpl(DeviceCand)->isRootDevice()) {
      try {
        return has_kernel(KernelIDCand,
                          DeviceCand.get_info<info::device::parent_device>());
      } catch (std::exception &e) {
        __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in has_kernel", e);
      }
    }
    return false;
  }

  iterator_range<std::vector<kernel_id>::const_iterator>
  get_kernel_ids() const noexcept {
    if (MKernelIDs)
      return *MKernelIDs;
    else
      return {};
  }
  // This should only be used when creating new device_image_impls that have the
  // exact same set of kernels as the source one. In all other scenarios the
  // getter above is the one needed:
  std::shared_ptr<std::vector<kernel_id>> &get_kernel_ids_ptr() noexcept {
    return MKernelIDs;
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
    assert(hasRTDeviceBinaryImage() &&
           "native_specialization_constant() called for unimplemented case");

    auto IsJITSPIRVTarget = [](const char *Target) {
      return (strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0 ||
              strcmp(Target, __SYCL_DEVICE_BINARY_TARGET_SPIRV32) == 0);
    };
    return (MContext.get_backend() == backend::opencl ||
            MContext.get_backend() == backend::ext_oneapi_level_zero) &&
           IsJITSPIRVTarget(get_bin_image_ref()->getRawData().DeviceTargetSpec);
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
    sycl_device_binary_property Prop =
        get_bin_image_ref()->getProperty("specConstsReplacedWithDefault");
    return Prop && (DeviceBinaryProperty(Prop).asUint32() != 0);
  }

  bundle_state get_state() const noexcept { return MState; }

  void set_state(bundle_state NewState) noexcept { MState = NewState; }

  devices_range get_devices() const noexcept { return MDevices; }

  bool compatible_with_device(device_impl &Dev) const {
    return get_devices().contains(Dev);
  }

  ur_program_handle_t get_ur_program() const noexcept { return MProgram; }

  const RTDeviceBinaryImage *const &get_bin_image_ref() const {
    return std::get<const RTDeviceBinaryImage *>(MBinImage);
  }

  const context &get_context() const noexcept { return MContext; }

  std::vector<unsigned char> &get_spec_const_blob_ref() noexcept {
    return MSpecConstsBlob;
  }

  ur_mem_handle_t &get_spec_const_buffer_ref() noexcept {
    std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
    if (nullptr == MSpecConstsBuffer && !MSpecConstsBlob.empty()) {
      adapter_impl &Adapter = getSyclObjImpl(MContext)->getAdapter();
      //  Uses UR_MEM_FLAGS_HOST_PTR_COPY instead of UR_MEM_FLAGS_HOST_PTR_USE
      //  since post-enqueue cleanup might trigger destruction of
      //  device_image_impl and, as a result, destruction of MSpecConstsBlob
      //  while MSpecConstsBuffer is still in use.
      //  TODO consider changing the lifetime of device_image_impl instead
      ur_buffer_properties_t Properties = {UR_STRUCTURE_TYPE_BUFFER_PROPERTIES,
                                           nullptr, MSpecConstsBlob.data()};
      try {
        memBufferCreateHelper(
            Adapter, detail::getSyclObjImpl(MContext)->getHandleRef(),
            UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER,
            MSpecConstsBlob.size(), &MSpecConstsBuffer, &Properties);
      } catch (std::exception &e) {
        __SYCL_REPORT_EXCEPTION_TO_STREAM(
            "exception in get_spec_const_buffer_ref", e);
      }
    }
    return MSpecConstsBuffer;
  }

  const SpecConstMapT &get_spec_const_data_ref() const noexcept {
    return MSpecConstSymMap;
  }

  std::mutex &get_spec_const_data_lock() noexcept {
    return MSpecConstAccessMtx;
  }

  ur_native_handle_t getNative() const {
    assert(MProgram);
    context_impl &ContextImpl = *detail::getSyclObjImpl(MContext);
    adapter_impl &Adapter = ContextImpl.getAdapter();

    ur_native_handle_t NativeProgram = 0;
    Adapter.call<UrApiKind::urProgramGetNativeHandle>(MProgram, &NativeProgram);
    if (ContextImpl.getBackend() == backend::opencl)
      __SYCL_OCL_CALL(clRetainProgram, ur::cast<cl_program>(NativeProgram));

    return NativeProgram;
  }

#ifdef _MSC_VER
#pragma warning(push)
// https://developercommunity.visualstudio.com/t/False-C4297-warning-while-using-function/1130300
// https://godbolt.org/z/xsMvKf84f
#pragma warning(disable : 4297)
#endif
  ~device_image_impl() try {
    if (MSpecConstsBuffer) {
      std::lock_guard<std::mutex> Lock{MSpecConstAccessMtx};
      adapter_impl &Adapter = getSyclObjImpl(MContext)->getAdapter();
      memReleaseHelper(Adapter, MSpecConstsBuffer);
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~device_image_impl", e);
    return; // Don't re-throw.
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  // Assumes the kernel is contained within this image.
  std::string_view getAdjustedKernelNameStrView(std::string_view Name) const {
    return getAdjustedKernelNameImpl<std::string_view>(Name);
  }

  std::string getAdjustedKernelNameStr(std::string_view Name) const {
    return getAdjustedKernelNameImpl<std::string>(Name);
  }

  bool hasKernelName(std::string_view Name) const {
    return (getOriginMask() &
            (ImageOriginKernelCompiler | ImageOriginSYCLBIN)) &&
           !Name.empty() &&
           MKernelNames.find(getAdjustedKernelNameStr(Name)) !=
               MKernelNames.end();
  }

  std::shared_ptr<kernel_impl>
  tryGetExtensionKernel(std::string_view Name, const context &Context,
                        const kernel_bundle_impl &OwnerBundle);

  bool hasDeviceGlobalName(const std::string &Name) const noexcept {
    if (!MRTCBinInfo.has_value())
      return false;

    return std::any_of(MRTCBinInfo->MDeviceGlobalRegistries.begin(),
                       MRTCBinInfo->MDeviceGlobalRegistries.end(),
                       [&Name](const auto &DGReg) {
                         return DGReg->hasDeviceGlobalName(Name);
                       });
  }

  DeviceGlobalMapEntry *tryGetDeviceGlobalEntry(const std::string &Name) const {
    if (!MRTCBinInfo.has_value())
      return nullptr;

    for (const auto &DGReg : MRTCBinInfo->MDeviceGlobalRegistries)
      if (DeviceGlobalMapEntry *DGEntry = DGReg->tryGetDeviceGlobalEntry(Name))
        return DGEntry;
    return nullptr;
  }

  uint8_t getOriginMask() const noexcept { return MOrigins; }

  const std::optional<KernelCompilerBinaryInfo> &getRTCInfo() const noexcept {
    return MRTCBinInfo;
  }

  const KernelNameSetT &getKernelNames() const noexcept { return MKernelNames; }

  const KernelNameToArgMaskMap &getEliminatedKernelArgMasks() const noexcept {
    return MEliminatedKernelArgMasks;
  }

  bool isNonSYCLSourceBased() const noexcept {
    return (getOriginMask() & ImageOriginKernelCompiler) &&
           !isFromSourceLanguage(syclex::source_language::sycl);
  }

  bool isFromSourceLanguage(syclex::source_language Lang) const noexcept {
    return MRTCBinInfo && MRTCBinInfo->MLanguage == Lang;
  }

  static ::jit_compiler::BinaryFormat
  getTargetFormat([[maybe_unused]] const backend Backend) {
#if SYCL_EXT_JIT_ENABLE
    switch (Backend) {
    case backend::ext_oneapi_level_zero:
    case backend::opencl:
      return ::jit_compiler::BinaryFormat::SPIRV;
    case backend::ext_oneapi_cuda:
      return ::jit_compiler::BinaryFormat::PTX;
    case backend::ext_oneapi_hip:
      return ::jit_compiler::BinaryFormat::AMDGCN;
    default:
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Backend does not support kernel_compiler extension");
    }
#else
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "JIT not supported");
#endif // SYCL_EXT_JIT_ENABLE
  }

  std::vector<std::shared_ptr<device_image_impl>> buildFromSource(
      devices_range Devices,
      const std::vector<sycl::detail::string_view> &BuildOptions,
      std::string *LogPtr,
      const std::vector<sycl::detail::string_view> &RegisteredKernelNames,
      std::vector<std::shared_ptr<ManagedDeviceBinaries>> &OutDeviceBins)
      const {
    assert(!std::holds_alternative<const RTDeviceBinaryImage *>(MBinImage));
    assert(MRTCBinInfo);
    assert(MOrigins & ImageOriginKernelCompiler);

    sycl::detail::context_impl &ContextImpl = *getSyclObjImpl(MContext);

    for (device_impl &Dev : Devices) {
      if (!ContextImpl.hasDevice(Dev)) {
        throw sycl::exception(make_error_code(errc::invalid),
                              "device not part of kernel_bundle context");
      }
      if (!Dev.extOneapiCanBuild(MRTCBinInfo->MLanguage)) {
        // This error cannot not be exercised in the current implementation, as
        // compatibility with a source language depends on the backend's
        // capabilities and all devices in one context share the same backend in
        // the current implementation, so this would lead to an error already
        // during construction of the source bundle.
        throw sycl::exception(make_error_code(errc::invalid),
                              "device does not support source language");
      }
    }

    if (MRTCBinInfo->MLanguage == syclex::source_language::sycl) {
      const auto Format = getTargetFormat(MContext.get_backend());
      return createSYCLImages(Devices, bundle_state::executable, BuildOptions,
                              LogPtr, RegisteredKernelNames, OutDeviceBins,
                              Format);
    }

    auto DeviceVec = Devices.to<std::vector<ur_device_handle_t>>();

    Managed<ur_program_handle_t> UrProgram;
    // SourceStrPtr will be null when source is Spir-V bytes.
    const std::string *SourceStrPtr = std::get_if<std::string>(&MBinImage);
    if (PersistentDeviceCodeCache::isEnabled() && SourceStrPtr) {
      UrProgram =
          extKernelCompilerFetchFromCache(Devices, BuildOptions, *SourceStrPtr);
    }
    bool FetchedFromCache = (UrProgram != nullptr);

    adapter_impl &Adapter = ContextImpl.getAdapter();

    if (!FetchedFromCache)
      UrProgram = createProgramFromSource(Devices, BuildOptions, LogPtr);

    std::string XsFlags = extractXsFlags(BuildOptions, MRTCBinInfo->MLanguage);
    auto Res = Adapter.call_nocheck<UrApiKind::urProgramBuildExp>(
        UrProgram, DeviceVec.size(), DeviceVec.data(), XsFlags.c_str());
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter.call_nocheck<UrApiKind::urProgramBuild>(
          ContextImpl.getHandleRef(), UrProgram, XsFlags.c_str());
    }
    Adapter.checkUrResult<errc::build>(Res);

    // Get the number of kernels in the program.
    size_t NumKernels;
    Adapter.call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_NUM_KERNELS, sizeof(size_t), &NumKernels,
        nullptr);

    std::vector<std::string> KernelNames =
        getKernelNamesFromURProgram(Adapter, UrProgram);
    KernelNameSetT KernelNameSet{KernelNames.begin(), KernelNames.end()};

    // If caching enabled and kernel not fetched from cache, cache.
    if (PersistentDeviceCodeCache::isEnabled() && !FetchedFromCache &&
        SourceStrPtr) {
      PersistentDeviceCodeCache::putCompiledKernelToDisc(
          Devices, syclex::detail::userArgsAsString(BuildOptions),
          *SourceStrPtr, UrProgram);
    }
    return std::vector<std::shared_ptr<device_image_impl>>{
        device_image_impl::create(MContext, Devices, bundle_state::executable,
                                  std::move(UrProgram), MRTCBinInfo->MLanguage,
                                  std::move(KernelNameSet))};
  }

  std::vector<std::shared_ptr<device_image_impl>> compileFromSource(
      devices_range Devices,
      const std::vector<sycl::detail::string_view> &CompileOptions,
      std::string *LogPtr,
      const std::vector<sycl::detail::string_view> &RegisteredKernelNames,
      std::vector<std::shared_ptr<ManagedDeviceBinaries>> &OutDeviceBins)
      const {
    assert(!std::holds_alternative<const RTDeviceBinaryImage *>(MBinImage));
    assert(MRTCBinInfo);
    assert(MOrigins & ImageOriginKernelCompiler);

    if (MRTCBinInfo->MLanguage != syclex::source_language::sycl)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "compile is only available for kernel_bundle<bundle_state::source> "
          "when the source language was sycl.");

    sycl::detail::context_impl &ContextImpl = *getSyclObjImpl(MContext);

    for (device_impl &Dev : Devices) {
      if (!ContextImpl.hasDevice(Dev)) {
        throw sycl::exception(make_error_code(errc::invalid),
                              "device not part of kernel_bundle context");
      }
      if (!Dev.extOneapiCanCompile(MRTCBinInfo->MLanguage)) {
        // This error cannot not be exercised in the current implementation, as
        // compatibility with a source language depends on the backend's
        // capabilities and all devices in one context share the same backend in
        // the current implementation, so this would lead to an error already
        // during construction of the source bundle.
        throw sycl::exception(make_error_code(errc::invalid),
                              "device does not support source language");
      }
    }
    const auto Format = getTargetFormat(MContext.get_backend());
    return createSYCLImages(Devices, bundle_state::object, CompileOptions,
                            LogPtr, RegisteredKernelNames, OutDeviceBins,
                            Format);
  }

private:
  template <typename RetT>
  RetT getAdjustedKernelNameImpl(std::string_view Name) const {
    if (MOrigins & ImageOriginSYCLBIN) {
      constexpr std::string_view KernelPrefix = "__sycl_kernel_";
      if (Name.size() > KernelPrefix.size() &&
          Name.substr(0, KernelPrefix.size()) == KernelPrefix)
        return RetT(Name);
      std::string AdjustedNameStr =
          std::string(KernelPrefix) + std::string(Name);
      if constexpr (std::is_same_v<RetT, std::string>) {
        return AdjustedNameStr;
      } else {
        static_assert(std::is_same_v<RetT, std::string_view>);
        auto It = MKernelNames.find(AdjustedNameStr);
        assert(It != MKernelNames.end() && "Adjusted name not found");
        return *It;
      }
    }

    if (!MRTCBinInfo.has_value())
      return RetT(Name);

    if (MRTCBinInfo->MLanguage == syclex::source_language::sycl) {
      auto It = MRTCBinInfo->MMangledKernelNames.find(Name);
      if (It != MRTCBinInfo->MMangledKernelNames.end())
        return It->second;
    }

    return RetT(Name);
  }

  bool hasRTDeviceBinaryImage() const noexcept {
    return std::holds_alternative<const RTDeviceBinaryImage *>(MBinImage) &&
           get_bin_image_ref() != nullptr;
  }

  static std::string_view trimXsFlags(std::string_view &str) {
    // Trim first and last quote if they exist, but no others.
    char EncounteredQuote = '\0';
    auto Start = std::find_if(str.begin(), str.end(), [&](char c) {
      if (!EncounteredQuote && (c == '\'' || c == '"')) {
        EncounteredQuote = c;
        return false;
      }
      return !std::isspace(c);
    });
    auto End = std::find_if(str.rbegin(), str.rend(), [&](char c) {
                 if (c == EncounteredQuote) {
                   EncounteredQuote = '\0';
                   return false;
                 }
                 return !std::isspace(c);
               }).base();
    if (Start != std::end(str) && End != std::begin(str) && Start < End) {
      return std::string_view(&*Start, std::distance(Start, End));
    }

    return "";
  }

  static std::string
  extractXsFlags(const std::vector<sycl::detail::string_view> &BuildOptions,
                 syclex::source_language lang) {
    std::stringstream SS;
    for (sycl::detail::string_view Option : BuildOptions) {
      if (lang == syclex::source_language::sycl) {
        std::string_view OptionSV{Option};
        auto Where = OptionSV.find("-Xs");
        if (Where != std::string_view::npos) {
          Where += 3;
          std::string_view Flags = OptionSV.substr(Where);
          SS << trimXsFlags(Flags) << " ";
        }
      } else {
        SS << std::string_view{Option} << " ";
      }
    }
    return SS.str();
  }

  Managed<ur_program_handle_t> extKernelCompilerFetchFromCache(
      devices_range Devices,
      const std::vector<sycl::detail::string_view> &BuildOptions,
      const std::string &SourceStr) const {
    sycl::detail::context_impl &ContextImpl = *getSyclObjImpl(MContext);
    adapter_impl &Adapter = ContextImpl.getAdapter();

    std::string UserArgs = syclex::detail::userArgsAsString(BuildOptions);

    auto DeviceHandles = Devices.to<std::vector<ur_device_handle_t>>();

    std::vector<const uint8_t *> Binaries;
    std::vector<size_t> Lengths;
    std::vector<std::vector<char>> BinProgs =
        PersistentDeviceCodeCache::getCompiledKernelFromDisc(Devices, UserArgs,
                                                             SourceStr);
    if (BinProgs.empty()) {
      return {};
    }
    for (auto &BinProg : BinProgs) {
      Binaries.push_back((uint8_t *)(BinProg.data()));
      Lengths.push_back(BinProg.size());
    }

    ur_program_properties_t Properties = {};
    Properties.stype = UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES;
    Properties.pNext = nullptr;
    Properties.count = 0;
    Properties.pMetadatas = nullptr;

    Managed<ur_program_handle_t> UrProgram{Adapter};
    Adapter.call<UrApiKind::urProgramCreateWithBinary>(
        ContextImpl.getHandleRef(), DeviceHandles.size(), DeviceHandles.data(),
        Lengths.data(), Binaries.data(), &Properties, &UrProgram);

    return UrProgram;
  }

  // Get the specialization constant default value blob.
  ByteArray getSpecConstsDefValBlob() const {
    if (!hasRTDeviceBinaryImage())
      return ByteArray(nullptr, 0);

    // Get default values for specialization constants.
    const RTDeviceBinaryImage::PropertyRange &SCDefValRange =
        get_bin_image_ref()->getSpecConstantsDefaultValues();
    if (!SCDefValRange.isAvailable())
      return ByteArray(nullptr, 0);

    ByteArray DefValDescriptors =
        DeviceBinaryProperty(*SCDefValRange.begin()).asByteArray();
    // First 8 bytes are consumed by the size of the property.
    DefValDescriptors.dropBytes(8);
    return DefValDescriptors;
  }

  void updateSpecConstSymMap() {
    if (hasRTDeviceBinaryImage()) {
      const RTDeviceBinaryImage::PropertyRange &SCRange =
          get_bin_image_ref()->getSpecConstants();
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

  std::vector<std::shared_ptr<device_image_impl>> createSYCLImages(
      devices_range Devices, bundle_state State,
      const std::vector<sycl::detail::string_view> &Options,
      std::string *LogPtr,
      const std::vector<sycl::detail::string_view> &RegisteredKernelNames,
      std::vector<std::shared_ptr<ManagedDeviceBinaries>> &OutDeviceBins,
      ::jit_compiler::BinaryFormat Format) const {
    assert(MRTCBinInfo);
    assert(MRTCBinInfo->MLanguage == syclex::source_language::sycl);
    assert(std::holds_alternative<std::string>(MBinImage));

    // Build device images via the program manager.
    const std::string &SourceStr = std::get<std::string>(MBinImage);
    std::ostringstream SourceExt;
    if (!RegisteredKernelNames.empty()) {
      SourceExt << SourceStr << '\n';

      auto EmitEntry =
          [&SourceExt](
              const sycl::detail::string_view &Name) -> std::ostringstream & {
        SourceExt << "  {\"" << Name.data() << "\", " << Name.data() << "}";
        return SourceExt;
      };

      SourceExt << "[[__sycl_detail__::__registered_kernels__(\n";
      for (auto It = RegisteredKernelNames.begin(),
                SecondToLast = RegisteredKernelNames.end() - 1;
           It != SecondToLast; ++It) {
        EmitEntry(*It) << ",\n";
      }
      EmitEntry(RegisteredKernelNames.back()) << "\n";
      SourceExt << ")]];\n";
    }

    auto [Binaries, Prefix] = syclex::detail::SYCL_JIT_Compile(
        RegisteredKernelNames.empty() ? SourceStr : SourceExt.str(),
        MRTCBinInfo->MIncludePairs, Options, LogPtr, Format);

    auto &PM = detail::ProgramManager::getInstance();

    // Add all binaries and keep the images for processing.
    std::vector<std::pair<RTDeviceBinaryImage *,
                          std::shared_ptr<std::vector<kernel_id>>>>
        NewImages;
    NewImages.reserve(Binaries->NumDeviceBinaries);
    for (int I = 0; I < Binaries->NumDeviceBinaries; I++) {
      sycl_device_binary Binary = &(Binaries->DeviceBinaries[I]);
      RTDeviceBinaryImage *NewImage = nullptr;
      auto KernelIDs = std::make_shared<std::vector<kernel_id>>();
      PM.addImage(Binary, /*RegisterImgExports=*/false, &NewImage,
                  KernelIDs.get());
      if (NewImage)
        NewImages.push_back(
            std::make_pair(std::move(NewImage), std::move(KernelIDs)));
    }

    // Now bring all images into the proper state. Note that we do this in a
    // separate pass over NewImages to make sure dependency images have been
    // registered beforehand.
    std::vector<std::shared_ptr<device_image_impl>> Result;
    Result.reserve(NewImages.size());
    for (auto &[NewImage, KernelIDs] : NewImages) {
      const RTDeviceBinaryImage &NewImageRef = *NewImage;

      // Filter the devices that support the image requirements.
      auto SupportingDevs = Devices.to<std::vector<device_impl *>>();
      auto NewSupportingDevsEnd = std::remove_if(
          SupportingDevs.begin(), SupportingDevs.end(),
          [&NewImageRef](device_impl *Dev) {
            return !doesDevSupportDeviceRequirements(*Dev, NewImageRef);
          });

      // If there are no devices that support the image, we skip it.
      if (NewSupportingDevsEnd == SupportingDevs.begin())
        continue;
      SupportingDevs.erase(NewSupportingDevsEnd, SupportingDevs.end());

      KernelNameSetT KernelNames;
      MangledKernelNameMapT MangledKernelNames;
      std::unordered_set<std::string> DeviceGlobalIDSet;
      std::vector<std::string> DeviceGlobalIDVec;
      std::vector<std::string> DeviceGlobalNames;
      std::vector<std::unique_ptr<std::byte[]>> DeviceGlobalAllocations;

      for (const auto &KernelID : *KernelIDs) {
        std::string_view KernelName{KernelID.get_name()};
        if (KernelName.find(Prefix) == 0) {
          KernelName.remove_prefix(Prefix.length());
          KernelNames.emplace(KernelName);
          static constexpr std::string_view SYCLKernelMarker{"__sycl_kernel_"};
          if (KernelName.find(SYCLKernelMarker) == 0) {
            // extern "C" declaration, implicitly register kernel without the
            // marker.
            std::string_view KernelNameWithoutMarker{KernelName};
            KernelNameWithoutMarker.remove_prefix(SYCLKernelMarker.length());
            MangledKernelNames.emplace(KernelNameWithoutMarker, KernelName);
          }
        }

        for (const sycl_device_binary_property &RKProp :
             NewImage->getRegisteredKernels()) {
          // Mangled names.
          auto BA = DeviceBinaryProperty(RKProp).asByteArray();
          auto MangledNameLen = BA.consume<uint64_t>() / 8 /*bits in a byte*/;
          std::string_view MangledName{
              reinterpret_cast<const char *>(BA.begin()), MangledNameLen};
          MangledKernelNames.emplace(RKProp->Name, MangledName);
        }

        // Device globals.
        for (const auto &DeviceGlobalProp : NewImage->getDeviceGlobals()) {
          std::string_view DeviceGlobalName{DeviceGlobalProp->Name};
          assert(DeviceGlobalName.find(Prefix) == 0);
          bool Inserted = false;
          std::tie(std::ignore, Inserted) =
              DeviceGlobalIDSet.emplace(DeviceGlobalName);
          if (Inserted) {
            DeviceGlobalIDVec.emplace_back(DeviceGlobalName);
            DeviceGlobalName.remove_prefix(Prefix.length());
            DeviceGlobalNames.emplace_back(DeviceGlobalName);
          }
        }
      }

      // Device globals are usually statically allocated and registered in the
      // integration footer, which we don't have in the RTC context. Instead,
      // we dynamically allocate storage tied to the executable kernel bundle.
      for (DeviceGlobalMapEntry *DeviceGlobalEntry :
           PM.getDeviceGlobalEntries(DeviceGlobalIDVec)) {

        size_t AllocSize = DeviceGlobalEntry->MDeviceGlobalTSize; // init value
        if (!DeviceGlobalEntry->MIsDeviceImageScopeDecorated) {
          // Consider storage for device USM pointer.
          AllocSize += sizeof(void *);
        }
        auto Alloc = std::make_unique<std::byte[]>(AllocSize);
        std::string_view DeviceGlobalName{DeviceGlobalEntry->MUniqueId};
        PM.addOrInitDeviceGlobalEntry(Alloc.get(), DeviceGlobalName.data());
        DeviceGlobalAllocations.push_back(std::move(Alloc));

        // Drop the RTC prefix from the entry's symbol name. Note that the PM
        // still manages this device global under its prefixed name.
        assert(DeviceGlobalName.find(Prefix) == 0);
        DeviceGlobalName.remove_prefix(Prefix.length());
        DeviceGlobalEntry->MUniqueId = DeviceGlobalName;
      }

      auto DGRegs = std::make_shared<ManagedDeviceGlobalsRegistry>(
          *getSyclObjImpl(MContext), std::string{Prefix},
          std::move(DeviceGlobalNames), std::move(DeviceGlobalAllocations));

      // Mark the image as input so the program manager will bring it into
      // the right state.
      auto DevImgImpl = device_image_impl::create(
          NewImage, MContext, SupportingDevs, bundle_state::input,
          std::move(KernelIDs), MRTCBinInfo->MLanguage, std::move(KernelNames),
          std::move(MangledKernelNames), std::string{Prefix},
          std::move(DGRegs));

      // Resolve dependencies.
      // If we are compiling to object, we do not want to error for unresolved
      // imports.
      // TODO: Consider making a collectDeviceImageDeps variant that takes a
      //       set reference and inserts into that instead.
      std::set<const RTDeviceBinaryImage *> ImgDeps;
      for (device_impl &Device : DevImgImpl->get_devices()) {
        std::set<const RTDeviceBinaryImage *> DevImgDeps =
            PM.collectDeviceImageDeps(*NewImage, Device,
                                      /*ErrorOnUnresolvableImport=*/State ==
                                          bundle_state::executable);
        ImgDeps.insert(DevImgDeps.begin(), DevImgDeps.end());
      }

      // Pack main image and dependencies together.
      std::vector<device_image_plain> NewImageAndDeps;
      NewImageAndDeps.reserve(
          1 + ((State == bundle_state::executable) * ImgDeps.size()));
      NewImageAndDeps.push_back(
          createSyclObjFromImpl<device_image_plain>(std::move(DevImgImpl)));
      devices_range SupportingDevsRef =
          getSyclObjImpl(NewImageAndDeps[0])->get_devices();
      if (State == bundle_state::executable) {
        // If target is executable we bundle the image and dependencies together
        // and bring it into state.
        for (const RTDeviceBinaryImage *ImgDep : ImgDeps)
          NewImageAndDeps.push_back(PM.createDependencyImage(
              MContext, SupportingDevsRef, ImgDep, bundle_state::input));
      } else if (State == bundle_state::object) {
        // If the target is object, we bring the dependencies into object state
        // individually and put them in the bundle.
        for (const RTDeviceBinaryImage *ImgDep : ImgDeps) {
          DevImgPlainWithDeps ImgDepWithDeps{PM.createDependencyImage(
              MContext, SupportingDevsRef, ImgDep, bundle_state::input)};
          PM.bringSYCLDeviceImageToState(ImgDepWithDeps, State);
          Result.push_back(getSyclObjImpl(ImgDepWithDeps.getMain()));
        }
      }

      DevImgPlainWithDeps ImgWithDeps(std::move(NewImageAndDeps));
      PM.bringSYCLDeviceImageToState(ImgWithDeps, State);
      Result.push_back(getSyclObjImpl(ImgWithDeps.getMain()));
    }

    OutDeviceBins.emplace_back(
        std::make_shared<ManagedDeviceBinaries>(std::move(Binaries)));
    return Result;
  }

  Managed<ur_program_handle_t>
  createProgramFromSource(devices_range Devices,
                          const std::vector<sycl::detail::string_view> &Options,
                          std::string *LogPtr) const {
    sycl::detail::context_impl &ContextImpl = *getSyclObjImpl(MContext);
    adapter_impl &Adapter = ContextImpl.getAdapter();
    const auto spirv = [&]() -> std::vector<uint8_t> {
      switch (MRTCBinInfo->MLanguage) {
      case syclex::source_language::opencl: {
        // if successful, the log is empty. if failed, throws an error with
        // the compilation log.
        const auto &SourceStr = std::get<std::string>(MBinImage);
        std::vector<uint32_t> IPVersionVec(Devices.size());
        std::transform(Devices.begin(), Devices.end(), IPVersionVec.begin(),
                       [&](device_impl &Dev) {
                         uint32_t ipVersion = 0;
                         Adapter.call<UrApiKind::urDeviceGetInfo>(
                             Dev.getHandleRef(), UR_DEVICE_INFO_IP_VERSION,
                             sizeof(uint32_t), &ipVersion, nullptr);
                         return ipVersion;
                       });
        return syclex::detail::OpenCLC_to_SPIRV(SourceStr, IPVersionVec,
                                                Options, LogPtr);
      }
      case syclex::source_language::spirv: {
        const auto &SourceBytes = std::get<std::vector<std::byte>>(MBinImage);
        std::vector<uint8_t> Result(SourceBytes.size());
        std::transform(SourceBytes.cbegin(), SourceBytes.cend(), Result.begin(),
                       [](std::byte B) { return static_cast<uint8_t>(B); });
        return Result;
      }
      default:
        break;
      }
      throw sycl::exception(
          make_error_code(errc::invalid),
          "SYCL C++, OpenCL C and SPIR-V are the only supported "
          "languages at this time");
    }();

    Managed<ur_program_handle_t> UrProgram{Adapter};
    Adapter.call<UrApiKind::urProgramCreateWithIL>(ContextImpl.getHandleRef(),
                                                   spirv.data(), spirv.size(),
                                                   nullptr, &UrProgram);
    if (UrProgram == nullptr)
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          "urProgramCreateWithIL resulted in a null program handle.");

    return UrProgram;
  }

  static std::vector<std::string>
  getKernelNamesFromURProgram(adapter_impl &Adapter,
                              ur_program_handle_t UrProgram) {
    // Get the kernel names.
    size_t KernelNamesSize;
    Adapter.call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_KERNEL_NAMES, 0u, nullptr, &KernelNamesSize);

    // semi-colon delimited list of kernel names.
    std::string KernelNamesStr(KernelNamesSize, ' ');
    Adapter.call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_KERNEL_NAMES, KernelNamesStr.size(),
        &KernelNamesStr[0], nullptr);
    return detail::split_string(KernelNamesStr, ';');
  }

  const std::variant<std::string, std::vector<std::byte>,
                     const RTDeviceBinaryImage *, SYCLBINBinaries>
      MBinImage = static_cast<const RTDeviceBinaryImage *>(nullptr);
  context MContext;
  std::vector<device_impl *> MDevices;
  bundle_state MState;
  // Native program handler which this device image represents
  Managed<ur_program_handle_t> MProgram;

  // List of kernel ids available in this image, elements should be sorted
  // according to LessByNameComp. Shared between images for performance reasons
  // (e.g. when we compile a single image it keeps the same kernels in it as the
  // original source image).
  std::shared_ptr<std::vector<kernel_id>> MKernelIDs;

  // List of known kernel names.
  KernelNameSetT MKernelNames;

  // Map for storing kernel argument masks for kernels. This is currently only
  // used for images created from SYCLBIN.
  KernelNameToArgMaskMap MEliminatedKernelArgMasks;

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
  ur_mem_handle_t MSpecConstsBuffer = nullptr;
  // Contains map of spec const names to their descriptions + offsets in
  // the MSpecConstsBlob
  std::map<std::string, std::vector<SpecConstDescT>> MSpecConstSymMap;

  // MOrigins is a bitfield to allow cases where the image is the product of
  // merging images of different origins.
  uint8_t MOrigins = ImageOriginSYCLOffline;
  // Optional information about the binary produced by the kernel compiler
  // extension.
  std::optional<KernelCompilerBinaryInfo> MRTCBinInfo = std::nullopt;

  // Used to store a dynamically created merged binary image, e.g. from linking.
  std::unique_ptr<DynRTDeviceBinaryImage> MMergedImageStorage = nullptr;
};

using device_images_iterator =
    variadic_iterator<device_image_plain,
                      std::vector<device_image_plain>::const_iterator,
                      std::set<device_image_impl *>::const_iterator>;
class device_images_range : public iterator_range<device_images_iterator> {
private:
  using Base = iterator_range<device_images_iterator>;

public:
  using Base::Base;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
