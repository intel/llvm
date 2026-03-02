//==---------------------- device_kernel_info.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <detail/device_kernel_info.hpp>
#include <detail/program_manager/program_manager.hpp>

#ifdef __has_include
#if __has_include(<cxxabi.h>)
#define __SYCL_ENABLE_GNU_DEMANGLING
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#endif
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceKernelInfo::DeviceKernelInfo(const CompileTimeKernelInfoTy &Info,
                                   std::optional<sycl::kernel_id> KernelID)
    : CompileTimeKernelInfoTy{Info}, MKernelID{std::move(KernelID)} {}

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
  assert(isCompileTimeInfoSet());
  assert(Info == *this);
}

void DeviceKernelInfo::setImplicitLocalArgPos(int Pos) {
  assert(!MImplicitLocalArgPos.has_value() || MImplicitLocalArgPos == Pos);
  MImplicitLocalArgPos = Pos;
}

std::string_view DeviceKernelInfo::getDemangledName() const {
  std::call_once(MDemangledNameInitFlag, [&]() {
#ifdef __SYCL_ENABLE_GNU_DEMANGLING
    int Status = -1; // some arbitrary value to eliminate the compiler warning
    char *Demangled =
        abi::__cxa_demangle(Name.data(), nullptr, nullptr, &Status);
    if (Status == 0 && Demangled) {
      std::unique_ptr<char, void (*)(void *)> Guard(Demangled, std::free);
      MDemangledName = std::string(Guard.get());
    } else {
      MDemangledName = std::string(Name);
    }
#else
    MDemangledName = std::string(Name);
#endif
  });
  return MDemangledName;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
