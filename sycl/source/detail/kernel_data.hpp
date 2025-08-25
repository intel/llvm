//==---------------- kernel_data.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/cg.hpp>
#include <detail/graph/dynamic_impl.hpp>

#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/kernel_name_based_cache.hpp>

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

  void *getKernelFuncPtr() const { return MKernelFuncPtr; }

  size_t getKernelNumArgs() const { return MKernelNumArgs; }

  KernelParamDescGetterT getKernelParamDescGetter() const {
    return MKernelParamDescGetter;
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // TODO: remove this method in the next ABI-breaking window
  //       it is used by handler code that will be removed in the next
  //       ABI-breaking window
  void setESIMD(bool IsESIMD) { MKernelIsESIMD = IsESIMD; }
#endif
  bool isESIMD() const { return MKernelIsESIMD; }

  bool hasSpecialCaptures() const { return MKernelHasSpecialCaptures; }

  KernelNameBasedCacheT *getKernelNameBasedCachePtr() const {
    return MKernelNameBasedCachePtr;
  }

  void setKernelNameBasedCachePtr(KernelNameBasedCacheT *Ptr) {
    MKernelNameBasedCachePtr = Ptr;
  }

  void setKernelInfo(void *KernelFuncPtr, int KernelNumArgs,
                     KernelParamDescGetterT KernelParamDescGetter,
                     bool KernelIsESIMD, bool KernelHasSpecialCaptures) {
    MKernelFuncPtr = KernelFuncPtr;
    MKernelNumArgs = KernelNumArgs;
    MKernelParamDescGetter = KernelParamDescGetter;
    MKernelIsESIMD = KernelIsESIMD;
    MKernelHasSpecialCaptures = KernelHasSpecialCaptures;
  }

  void processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                  const int Size, const size_t Index, size_t &IndexShift,
                  bool IsKernelCreatedFromSource, const NDRDescT &NDRDesc);

  void extractArgsAndReqs(const NDRDescT &NDRDesc,
                          bool IsKernelCreatedFromSource);

  void extractArgsAndReqsFromLambda(const NDRDescT &NDRDesc);

private:
  // Storage for any SYCL Graph dynamic parameters which have been flagged for
  // registration in the CG, along with the argument index for the parameter.
  DynamicParametersVecT MDynamicParameters;

  /// The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;

  // Store information about the kernel arguments.
  void *MKernelFuncPtr = nullptr;
  size_t MKernelNumArgs = 0;
  KernelParamDescGetterT MKernelParamDescGetter = nullptr;
  bool MKernelIsESIMD = false;
  bool MKernelHasSpecialCaptures = true;

  // A pointer to a kernel name based cache retrieved on the application side.
  KernelNameBasedCacheT *MKernelNameBasedCachePtr = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
