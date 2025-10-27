//==---------------- kernel_data.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_impl.hpp>
#include <detail/device_kernel_info.hpp>
#include <detail/graph/dynamic_impl.hpp>
#include <detail/kernel_arg_desc.hpp>
#include <detail/ndrange_desc.hpp>

#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/kernel_launch_helper.hpp>

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
  template <typename T>
  using MarshalledProperty = detail::MarshalledProperty<T>;
  template <int T>
  using cluster_size_key =
      sycl::ext::oneapi::experimental::cuda::cluster_size_key<T>;

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

  // Kernel launch properties getter and setters.
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

  void parseAndSetCacheConfigProperty(
      const sycl::ext::intel::experimental::cache_config_key &prop) {
    using namespace sycl::ext::intel::experimental;

    ur_kernel_cache_config_t CacheConfig =
        ur_kernel_cache_config_t::UR_KERNEL_CACHE_CONFIG_DEFAULT;
    if (prop == large_slm) {
      CacheConfig = ur_kernel_cache_config_t::UR_KERNEL_CACHE_CONFIG_LARGE_SLM;
    } else if (prop == large_data) {
      CacheConfig = ur_kernel_cache_config_t::UR_KERNEL_CACHE_CONFIG_LARGE_DATA;
    } else
      assert(false && "unknown cache property type");

    MKernelCacheConfig = CacheConfig;
  }

  template <int ClusterDims>
  void parseAndSetClusterDimProperty(
      const std::optional<cluster_size_key<ClusterDims>> &prop) {
    if (prop) {
      static_assert(ClusterDims < 4 && ClusterDims > 0,
                    "Invalid cluster dimensions");

      auto ClusterSize = prop->get_cluster_size();
      MKernelUsesClusterLaunch = true;

      if constexpr (ClusterDims == 1)
        MNDRDesc.setClusterDimensions(sycl::range<1>{ClusterSize[0]});
      else if constexpr (ClusterDims == 2)
        MNDRDesc.setClusterDimensions(
            sycl::range<2>{ClusterSize[0], ClusterSize[1]});
      else if constexpr (ClusterDims == 3)
        MNDRDesc.setClusterDimensions(
            sycl::range<3>{ClusterSize[0], ClusterSize[1], ClusterSize[2]});
    }
  }

  void validateAndSetKernelLaunchProperties(
      const detail::KernelPropertyHolderStructTy Kprop, bool HasGraph,
      const device_impl &dev) {
    using execScope = ext::oneapi::experimental::execution_scope;
    using namespace sycl::ext::oneapi::experimental;
    using namespace sycl::ext::oneapi::experimental::detail;

    const auto *WorkGroupMemSizeProp =
        static_cast<const MarshalledProperty<work_group_scratch_size> *>(
            &Kprop);
    const auto *CacheConfigProp = static_cast<const MarshalledProperty<
        sycl::ext::intel::experimental::cache_config_key> *>(&Kprop);
    const auto *UseRootSyncProp =
        static_cast<const MarshalledProperty<use_root_sync_key> *>(&Kprop);
    const auto *ForwardProgressProp =
        static_cast<const MarshalledProperty<work_group_progress_key> *>(
            &Kprop);
    const auto *ClusterLaunchPropDim1 =
        static_cast<const MarshalledProperty<cuda::cluster_size_key<1>> *>(
            &Kprop);
    const auto *ClusterLaunchPropDim2 =
        static_cast<const MarshalledProperty<cuda::cluster_size_key<2>> *>(
            &Kprop);
    const auto *ClusterLaunchPropDim3 =
        static_cast<const MarshalledProperty<cuda::cluster_size_key<3>> *>(
            &Kprop);

    const bool isClusterDimPropPresent = ClusterLaunchPropDim1->property ||
                                         ClusterLaunchPropDim2->property ||
                                         ClusterLaunchPropDim3->property;

    // Early validation for graph-incompatible properties
    if (HasGraph) {
      if (WorkGroupMemSizeProp->property) {
        throw sycl::exception(
            sycl::make_error_code(errc::invalid),
            "Setting work group scratch memory size is not yet supported "
            "for use with the SYCL Graph extension.");
      }

      if (isClusterDimPropPresent) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "Cluster launch is not yet supported "
                              "for use with the SYCL Graph extension.");
      }
    }

    // Validate and set forward progress guarantees.
    for (int i = 0; i < 3; i++) {
      if (ForwardProgressProp->MForwardProgressProperties[i]
              .Guarantee.has_value()) {

        if (!dev.supportsForwardProgress(
                *ForwardProgressProp->MForwardProgressProperties[i].Guarantee,
                *ForwardProgressProp->MForwardProgressProperties[i].ExecScope,
                *ForwardProgressProp->MForwardProgressProperties[i]
                     .CoordinationScope)) {
          throw sycl::exception(
              sycl::make_error_code(errc::feature_not_supported),
              "The device associated with the queue does not support the "
              "requested forward progress guarantee.");
        }

        auto execScope =
            *ForwardProgressProp->MForwardProgressProperties[i].ExecScope;
        // If we are here, the device supports the guarantee required but
        // there is a caveat in that if the guarantee required is a concurrent
        // guarantee, then we most likely also need to enable cooperative
        // launch of the kernel. That is, although the device supports the
        // required guarantee, some setup work is needed to truly make the
        // device provide that guarantee at runtime. Otherwise, we will get
        // the default guarantee which is weaker than concurrent. Same
        // reasoning applies for sub_group but not for work_item.
        // TODO: Further design work is probably needed to reflect this
        // behavior in Unified Runtime.
        if ((execScope == execScope::work_group ||
             execScope == execScope::sub_group) &&
            (*ForwardProgressProp->MForwardProgressProperties[i].Guarantee ==
             forward_progress_guarantee::concurrent)) {
          setCooperative(true);
        }
      }
    }

    if (UseRootSyncProp->isRootSyncPropPresent)
      setCooperative(true);

    if (CacheConfigProp->property)
      parseAndSetCacheConfigProperty(*(CacheConfigProp->property));

    if (WorkGroupMemSizeProp->property)
      setKernelWorkGroupMemorySize((*WorkGroupMemSizeProp->property).size);

    if (isClusterDimPropPresent) {
      parseAndSetClusterDimProperty(ClusterLaunchPropDim1->property);
      parseAndSetClusterDimProperty(ClusterLaunchPropDim2->property);
      parseAndSetClusterDimProperty(ClusterLaunchPropDim3->property);
    }
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
