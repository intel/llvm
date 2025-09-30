//==---------------- kernel_data.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_kernel_info.hpp>
#include <detail/graph/dynamic_impl.hpp>
#include <detail/kernel_arg_desc.hpp>
#include <detail/ndrange_desc.hpp>

#include <sycl/detail/kernel_desc.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class KernelData {
public:
  using KernelParamDescGetterT = detail::kernel_param_desc_t (*)(int);
  using DynamicParametersVecT = std::vector<std::pair<
      ext::oneapi::experimental::detail::dynamic_parameter_impl *, int>>;
  using ArgsVecT = std::vector<detail::ArgDesc>;

  KernelData() = default;
  ~KernelData() = default;
  KernelData(const KernelData &) = default;
  KernelData(KernelData &&) = default;
  KernelData &operator=(const KernelData &) = default;
  KernelData &operator=(KernelData &&) = default;

  DynamicParametersVecT &getDynamicParameters() { return MDynamicParameters; }

  const DynamicParametersVecT &getDynamicParameters() const {
    return MDynamicParameters;
  }

  template <typename... Args> void addDynamicParameter(Args &&...args) {
    MDynamicParameters.emplace_back(std::forward<Args>(args)...);
  }

  ArgsVecT &getArgs() & { return MArgs; }

  const ArgsVecT &getArgs() const & { return MArgs; }

  ArgsVecT &&getArgs() && { return std::move(MArgs); }

  void setArgs(const ArgsVecT &Args) { MArgs = Args; }

  void addArg(const detail::ArgDesc &Arg) { MArgs.push_back(Arg); }

  template <typename... Args> void addArg(Args &&...args) {
    MArgs.emplace_back(std::forward<Args>(args)...);
  }

  void clearArgs() { MArgs.clear(); }

  detail::NDRDescT &getNDRDesc() & { return MNDRDesc; }

  const detail::NDRDescT &getNDRDesc() const & { return MNDRDesc; }

  detail::NDRDescT &&getNDRDesc() && { return std::move(MNDRDesc); }

  void setNDRDesc(const detail::NDRDescT &NDRDesc) { MNDRDesc = NDRDesc; }

  void *getKernelFuncPtr() const { return MKernelFuncPtr; }

  size_t getKernelNumArgs() const {
    assert(MDeviceKernelInfoPtr);
    return MDeviceKernelInfoPtr->NumParams;
  }

  KernelParamDescGetterT getKernelParamDescGetter() const {
    assert(MDeviceKernelInfoPtr);
    return MDeviceKernelInfoPtr->ParamDescGetter;
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // TODO: remove this method in the next ABI-breaking window
  //       it is used by handler code that will be removed in the next
  //       ABI-breaking window
  void setESIMD(bool IsESIMD) {
    assert(MDeviceKernelInfoPtr);
    MDeviceKernelInfoPtr->IsESIMD = IsESIMD;
  }
#endif
  bool isESIMD() const {
    assert(MDeviceKernelInfoPtr);
    return MDeviceKernelInfoPtr->IsESIMD;
  }

  bool hasSpecialCaptures() const {
    assert(MDeviceKernelInfoPtr);
    return MDeviceKernelInfoPtr->HasSpecialCaptures;
  }

  DeviceKernelInfo *getDeviceKernelInfoPtr() const {
    return MDeviceKernelInfoPtr;
  }

  void setDeviceKernelInfoPtr(DeviceKernelInfo *Ptr) {
    MDeviceKernelInfoPtr = Ptr;
  }
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  void setKernelInfo(void *KernelFuncPtr, int KernelNumArgs,
                     KernelParamDescGetterT KernelParamDescGetter,
                     bool KernelIsESIMD, bool KernelHasSpecialCaptures) {
    MKernelFuncPtr = KernelFuncPtr;

    assert(MDeviceKernelInfoPtr &&
           "MDeviceKernelInfoPtr must be set before calling setKernelInfo");

    detail::CompileTimeKernelInfoTy Info;
    Info.Name = MDeviceKernelInfoPtr->Name;
    Info.NumParams = KernelNumArgs;
    Info.ParamDescGetter = KernelParamDescGetter;
    Info.IsESIMD = KernelIsESIMD;
    Info.HasSpecialCaptures = KernelHasSpecialCaptures;

    MDeviceKernelInfoPtr->initIfEmpty(Info);
  }
#endif

  void setKernelFunc(void *KernelFuncPtr) { MKernelFuncPtr = KernelFuncPtr; }

  bool usesAssert() const {
    assert(MDeviceKernelInfoPtr);
    return MDeviceKernelInfoPtr->usesAssert();
  }

  ur_kernel_cache_config_t getKernelCacheConfig() const {
    return MKernelCacheConfig;
  }

  void setKernelCacheConfig(ur_kernel_cache_config_t Config) {
    MKernelCacheConfig = Config;
  }

  bool isCooperative() const { return MKernelIsCooperative; }

  void setCooperative(bool IsCooperative) {
    MKernelIsCooperative = IsCooperative;
  }

  bool usesClusterLaunch() const { return MKernelUsesClusterLaunch; }

  template <int Dims_> void setClusterDimensions(sycl::range<Dims_> N) {
    MKernelUsesClusterLaunch = true;
    MNDRDesc.setClusterDimensions(N);
  }

  uint32_t getKernelWorkGroupMemorySize() const {
    return MKernelWorkGroupMemorySize;
  }

  void setKernelWorkGroupMemorySize(uint32_t Size) {
    MKernelWorkGroupMemorySize = Size;
  }

  KernelNameStrRefT getKernelName() const {
    assert(MDeviceKernelInfoPtr);
    return static_cast<KernelNameStrRefT>(MDeviceKernelInfoPtr->Name);
  }

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
                  ,
                  bool IsESIMD
#endif
  );

  void extractArgsAndReqs(bool IsKernelCreatedFromSource);

  void extractArgsAndReqsFromLambda();

private:
  // Storage for any SYCL Graph dynamic parameters which have been flagged for
  // registration in the CG, along with the argument index for the parameter.
  DynamicParametersVecT MDynamicParameters;

  /// The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;

  ur_kernel_cache_config_t MKernelCacheConfig = UR_KERNEL_CACHE_CONFIG_DEFAULT;

  bool MKernelIsCooperative = false;
  bool MKernelUsesClusterLaunch = false;
  uint32_t MKernelWorkGroupMemorySize = 0;

  /// Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;

  // Store information about the kernel arguments.
  void *MKernelFuncPtr = nullptr;

  // A pointer to device kernel information. Cached on the application side in
  // headers or retrieved from program manager.
  DeviceKernelInfo *MDeviceKernelInfoPtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
