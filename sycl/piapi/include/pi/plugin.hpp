//==--------------------- plugin.hpp - SYCL platform-------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <pi/pi.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti_trace_framework.h"
#endif

#define PI_CHECK_OCL_CODE_THROW_HELPER(X, EXC)                                 \
  if (X != 0) {                                                                \
    throw EXC;                                                                 \
  }

#ifdef PI_DPCPP_INTEGRATION
#include <CL/sycl/detail/common.hpp>
#else
#ifndef __SYCL_CHECK_OCL_CODE_THROW
#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC)                                    \
  PI_CHECK_OCL_CODE_THROW_HELPER(X, EXC{})
#endif // __SYCL_CHECK_OCL_CODE_THROW
#endif // PI_DPCPP_INTEGRATION

namespace pi {
/// The plugin class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_pi
class plugin {
public:
  plugin() = delete;

  plugin(pi::PiPlugin Plugin, backend UseBackend)
      : MPlugin(Plugin), MBackend(UseBackend) {}

  plugin &operator=(const plugin &) = default;
  plugin(const plugin &) = default;
  plugin &operator=(plugin &&other) noexcept = default;
  plugin(plugin &&other) noexcept = default;

  ~plugin() = default;

  const pi::PiPlugin &getPiPlugin() const { return MPlugin; }
  pi::PiPlugin &getPiPlugin() { return MPlugin; }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = std::runtime_error>
  void checkPiResult(pi::PiResult pi_result) const {
    __SYCL_CHECK_OCL_CODE_THROW(pi_result, Exception);
  }

  /// Calls the PiApi, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// PiResult Err = plugin.call<PiApiKind::pi>(Args);
  /// Plugin.checkPiResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa plugin::checkPiResult
  template <PiApiKind PiApiOffset, typename... ArgsT>
  pi::PiResult call_nocheck(ArgsT... Args) const {
    pi::PiFuncInfo<PiApiOffset> PiCallInfo;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Emit a function_begin trace for the PI API before the call is executed.
    // If arguments need to be captured, then a data structure can be sent in
    // the per_instance_user_data field.
    std::string PIFnName = PiCallInfo.getFuncName();
    uint64_t CorrelationID = pi::emitFunctionBeginTrace(PIFnName.c_str());
#endif
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::string FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      pi::printArgs(Args...);
    }
    pi::PiResult R = PiCallInfo.getFuncPtr(MPlugin)(Args...);
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::cout << ") ---> ";
      pi::printArgs(R);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Close the function begin with a call to function end
    pi::emitFunctionEndTrace(CorrelationID, PIFnName.c_str());
#endif
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw cl::sycl::runtime_exception if the call was not successful.
  template <PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    pi::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult(Err);
  }

  backend getBackend(void) const { return MBackend; }

private:
  pi::PiPlugin MPlugin;
  backend MBackend;
}; // class plugin

template <>
inline void
plugin::checkPiResult<std::runtime_error>(pi::PiResult pi_result) const {
  PI_CHECK_OCL_CODE_THROW_HELPER(pi_result,
                                 std::runtime_error{"Invalid PIAPI call"});
}

// Holds the PluginInformation for the plugin that is bound.
// Currently a global variable is used to store OpenCL plugin information to be
// used with SYCL Interoperability Constructors.
extern std::shared_ptr<pi::plugin> GlobalPlugin;

} // namespace pi
