//==--------------------- plugin.hpp - SYCL platform-------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/stl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class plugin {
public:
  plugin() = delete;

  plugin(RT::PiPlugin Plugin) : MPlugin(Plugin) {
    MPiEnableTrace = (std::getenv("SYCL_PI_TRACE") != nullptr);
  }

  ~plugin() = default;

  // Utility function to check return from PI calls.
  // Throws if pi_result is not a PI_SUCCESS.
  // Exception - The type of exception to throw if PiResult of a call is not
  // PI_SUCCESS. Default value is cl::sycl::runtime_error.
  template <typename Exception = cl::sycl::runtime_error>
  void checkPiResult(RT::PiResult pi_result) const {
    CHECK_OCL_CODE_THROW(pi_result, Exception);
  }

  // Call the PiApi, trace the call and return the result.
  // To check the result use checkPiResult.
  // Usage:
  // PiResult Err = plugin.call<PiApiKind::pi>(Args);
  // Plugin.checkPiResult(Err); <- Checks Result and throws a runtime_error
  // exception.
  template <PiApiKind PiApiOffset, typename... ArgsT>
  RT::PiResult call_nocheck(ArgsT... Args) const {
    RT::PiFuncInfo<PiApiOffset> PiCallInfo;
    if (MPiEnableTrace) {
      std::string FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      RT::printArgs(Args...);
    }
    RT::PiResult R = PiCallInfo.getFuncPtr(MPlugin)(Args...);
    if (MPiEnableTrace) {
      std::cout << ") ---> ";
      RT::printArgs(R);
    }
    return R;
  }

  // Call the API, trace the call, check the result and throw
  // a runtime_error Exception
  template <PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    RT::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult(Err);
  }
  // TODO: Make this private. Currently used in program_manager to create a
  // pointer to PiProgram.
  RT::PiPlugin MPlugin;

private:
  bool MPiEnableTrace;

}; // class plugin
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
