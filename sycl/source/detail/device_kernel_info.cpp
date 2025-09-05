//==---------------------- device_kernel_info.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/device_kernel_info.hpp>
#include <detail/program_manager/program_manager.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceKernelInfo::DeviceKernelInfo(const CompileTimeKernelInfoTy &Info)
    : CompileTimeKernelInfoTy(Info)
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
      ,
      Name(Info.Name.data())
#endif
{
  init(Name.data());
}

void DeviceKernelInfo::init(KernelNameStrRefT KernelName) {
  auto &PM = detail::ProgramManager::getInstance();
  MUsesAssert = PM.kernelUsesAssert(KernelName);
  MImplicitLocalArgPos = PM.kernelImplicitLocalArgPos(KernelName);
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  MInitialized.store(true);
#endif
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void DeviceKernelInfo::initIfEmpty(const CompileTimeKernelInfoTy &Info) {
  if (MInitialized.load())
    return;

  CompileTimeKernelInfoTy::operator=(Info);
  Name = Info.Name.data();
  init(Name.data());
}
#endif

template <typename OtherTy>
inline constexpr bool operator==(const CompileTimeKernelInfoTy &LHS,
                                 const OtherTy &RHS) {
  // TODO replace with std::tie(...) == std::tie(...) once there is
  // implicit conversion from detail to std string_view.
  return std::string_view{LHS.Name} == std::string_view{RHS.Name} &&
         LHS.NumParams == RHS.NumParams && LHS.IsESIMD == RHS.IsESIMD &&
         std::string_view{LHS.FileName} == std::string_view{RHS.FileName} &&
         std::string_view{LHS.FunctionName} ==
             std::string_view{RHS.FunctionName} &&
         LHS.LineNumber == RHS.LineNumber &&
         LHS.ColumnNumber == RHS.ColumnNumber &&
         LHS.KernelSize == RHS.KernelSize &&
         LHS.ParamDescGetter == RHS.ParamDescGetter &&
         LHS.HasSpecialCaptures == RHS.HasSpecialCaptures;
}

void DeviceKernelInfo::setCompileTimeInfoIfNeeded(
    const CompileTimeKernelInfoTy &Info) {
  if (isCompileTimeInfoSet())
    CompileTimeKernelInfoTy::operator=(Info);
  assert(isCompileTimeInfoSet());
  assert(Info == *this);
}

FastKernelSubcacheT &DeviceKernelInfo::getKernelSubcache() {
  assertInitialized();
  return MFastKernelSubcache;
}
bool DeviceKernelInfo::usesAssert() {
  assertInitialized();
  return MUsesAssert;
}
const std::optional<int> &DeviceKernelInfo::getImplicitLocalArgPos() {
  assertInitialized();
  return MImplicitLocalArgPos;
}

bool DeviceKernelInfo::isCompileTimeInfoSet() const { return KernelSize != 0; }

void DeviceKernelInfo::assertInitialized() {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  assert(MInitialized.load() && "Data needs to be initialized before use");
#endif
}

} // namespace detail
} // namespace _V1
} // namespace sycl
