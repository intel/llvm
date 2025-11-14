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
    : CompileTimeKernelInfoTy(Info) {
  init(Name.data());
}

void DeviceKernelInfo::init(KernelNameStrRefT KernelName) {
  auto &PM = detail::ProgramManager::getInstance();
  MUsesAssert = PM.kernelUsesAssert(KernelName);
  MImplicitLocalArgPos = PM.kernelImplicitLocalArgPos(KernelName);
}

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
  // In case of 6.3 compatibility mode the KernelSize is not passed to the
  // runtime. So, it will always be 0 and this assert fails.
  assert(isCompileTimeInfoSet());
  assert(Info == *this);
}

FastKernelSubcacheT &DeviceKernelInfo::getKernelSubcache() {
  return MFastKernelSubcache;
}
bool DeviceKernelInfo::usesAssert() { return MUsesAssert; }
const std::optional<int> &DeviceKernelInfo::getImplicitLocalArgPos() {
  return MImplicitLocalArgPos;
}

bool DeviceKernelInfo::isCompileTimeInfoSet() const { return KernelSize != 0; }

} // namespace detail
} // namespace _V1
} // namespace sycl
