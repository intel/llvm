//==--------------------- plugin.hpp - SYCL platform-------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/plugin_printers.hpp>
#include <memory>
#include <mutex>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GPICallEvent;
#endif
/// The plugin class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_pi
class plugin {
public:
  plugin() = delete;

  plugin(RT::PiPlugin Plugin, backend UseBackend, void *LibraryHandle)
      : MPlugin(Plugin), MBackend(UseBackend), MLibraryHandle(LibraryHandle),
        TracingMutex(std::make_shared<std::mutex>()) {}

  plugin &operator=(const plugin &) = default;
  plugin(const plugin &) = default;
  plugin &operator=(plugin &&other) noexcept = default;
  plugin(plugin &&other) noexcept = default;

  ~plugin() = default;

  const RT::PiPlugin &getPiPlugin() const { return MPlugin; }
  RT::PiPlugin &getPiPlugin() { return MPlugin; }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = cl::sycl::runtime_error>
  void checkPiResult(RT::PiResult pi_result) const {
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
  RT::PiResult call_nocheck(ArgsT... Args) const {
    RT::PiFuncInfo<PiApiOffset> PiCallInfo;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Emit a function_begin trace for the PI API before the call is executed.
    // If arguments need to be captured, then a data structure can be sent in
    // the per_instance_user_data field.
    std::string PIFnName = PiCallInfo.getFuncName();
    uint64_t CorrelationID = pi::emitFunctionBeginTrace(PIFnName.c_str());
#endif
    RT::PiResult R;
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::lock_guard<std::mutex> Guard(*TracingMutex);
      std::string FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      RT::printArgs(Args...);
      R = PiCallInfo.getFuncPtr(MPlugin)(Args...);
      std::cout << ") ---> ";
      RT::printArgs(R);
      RT::printOuts(Args...);
      std::cout << std::endl;
    } else {
      R = PiCallInfo.getFuncPtr(MPlugin)(Args...);
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
    RT::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult(Err);
  }

  backend getBackend(void) const { return MBackend; }
  void *getLibraryHandle() const { return MLibraryHandle; }
  void *getLibraryHandle() { return MLibraryHandle; }
  int unload() { return RT::unloadPlugin(MLibraryHandle); }

private:
  RT::PiPlugin MPlugin;
  backend MBackend;
  void *MLibraryHandle; // the handle returned from dlopen
  std::shared_ptr<std::mutex> TracingMutex;
}; // class plugin
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
