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
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  // Non-legacy implementation either fills out the data during image
  // registration after this constructor is called, or uses default values
  // if this instance of DeviceKernelInfo corresponds to an interop kernel.
  MInitialized.store(true);
#endif
}

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
void DeviceKernelInfo::initIfEmpty(const CompileTimeKernelInfoTy &Info) {
  if (MInitialized.load())
    return;

  // If this function is called, then this is a default initialized
  // device kernel info created from older headers and stored in global handler.
  // In that case, fetch the proper instance from program manager and copy its
  // values.
  auto &PM = detail::ProgramManager::getInstance();
  DeviceKernelInfo &PMDeviceKernelInfo =
      PM.getDeviceKernelInfo(KernelNameStrRefT(Info.Name));

  PMDeviceKernelInfo.CompileTimeKernelInfoTy::operator=(Info);
  PMDeviceKernelInfo.Name = Info.Name.data();

  MUsesAssert = PMDeviceKernelInfo.MUsesAssert;
  MImplicitLocalArgPos = PMDeviceKernelInfo.MImplicitLocalArgPos;
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
         // TODO This check fails with test_handler CTS due to what appears to
         // be a test bug. Disable it for now as a workaround.
         // See https://github.com/intel/llvm/issues/20134 for more info.
         // LHS.ParamDescGetter == RHS.ParamDescGetter &&
         LHS.HasSpecialCaptures == RHS.HasSpecialCaptures;
}

void DeviceKernelInfo::setCompileTimeInfoIfNeeded(
    const CompileTimeKernelInfoTy &Info) {
  if (!isCompileTimeInfoSet())
    CompileTimeKernelInfoTy::operator=(Info);
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // In case of 6.3 compatibility mode the KernelSize is not passed to the
  // runtime. So, it will always be 0 and this assert fails.
  assert(isCompileTimeInfoSet());
#endif
  assert(Info == *this);
}

FastKernelSubcacheT &DeviceKernelInfo::getKernelSubcache() {
  assertInitialized();
  return MFastKernelSubcache;
}
bool DeviceKernelInfo::usesAssert() const {
  assertInitialized();
  return MUsesAssert;
}
const std::optional<int> &DeviceKernelInfo::getImplicitLocalArgPos() const {
  assertInitialized();
  return MImplicitLocalArgPos;
}

void DeviceKernelInfo::setUsesAssert() { MUsesAssert = true; }

void DeviceKernelInfo::setImplicitLocalArgPos(int Pos) {
  assert(!MImplicitLocalArgPos.has_value() || MImplicitLocalArgPos == Pos);
  MImplicitLocalArgPos = Pos;
}

bool DeviceKernelInfo::isCompileTimeInfoSet() const { return KernelSize != 0; }

void DeviceKernelInfo::assertInitialized() const {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  assert(MInitialized.load() && "Data needs to be initialized before use");
#endif
}

} // namespace detail
} // namespace _V1
} // namespace sycl
