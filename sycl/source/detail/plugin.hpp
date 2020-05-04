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

  plugin(RT::PiPlugin Plugin, backend UseBackend)
      : MPlugin(Plugin), MBackend(UseBackend) {}

  ~plugin() = default;

  const RT::PiPlugin &getPiPlugin() const { return MPlugin; }
  RT::PiPlugin &getPiPlugin() { return MPlugin; }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = cl::sycl::runtime_error>
  void checkPiResult(RT::PiResult pi_result) const {
    CHECK_OCL_CODE_THROW(pi_result, Exception);
  }

  uint64_t emitFunctionBeginTrace(const char *FName) const {
    uint64_t CorrelationID = 0;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // The function_begin and function_end trace point types are defined to
    // trace library API calls and they are currently enabled here for supports
    // tools that need the API scope. The methods emitFunctionBeginTrace() and
    // emitFunctionEndTrace() can be extended to also trace the arguments of the
    // PI API call using a trace point type the extends the predefined trace
    // point types.
    //
    // You can use the sample collector in llvm/xptifw/samples/syclpi_collector
    // to print the API traces and also extend them to support an arguments that
    // may be traced later.
    if (xptiTraceEnabled()) {
      uint8_t StreamID = xptiRegisterStream(SYCL_PICALL_STREAM_NAME);
      CorrelationID = xptiGetUniqueId();
      xptiNotifySubscribers(StreamID,
                            (uint16_t)xpti::trace_point_type_t::function_begin,
                            GPICallEvent, nullptr, CorrelationID,
                            static_cast<const void *>(FName));
    }
#endif
    return CorrelationID;
  }

  void emitFunctionEndTrace(uint64_t CorrelationID, const char *FName) const {
#ifdef XPTI_ENABLE_INSTRUMENTATION
    if (xptiTraceEnabled()) {
      // CorrelationID is the unique ID that ties together a function_begin and
      // function_end pair of trace calls. The splitting of a scoped_notify into
      // two function calls incurs an additional overhead as the StreamID must
      // be looked up twice.
      uint8_t StreamID = xptiRegisterStream(SYCL_PICALL_STREAM_NAME);
      xptiNotifySubscribers(StreamID,
                            (uint16_t)xpti::trace_point_type_t::function_end,
                            GPICallEvent, nullptr, CorrelationID,
                            static_cast<const void *>(FName));
    }
#endif
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
    uint64_t CorrelationID = emitFunctionBeginTrace(PIFnName.c_str());
#endif
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::string FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      RT::printArgs(Args...);
    }
    RT::PiResult R = PiCallInfo.getFuncPtr(MPlugin)(Args...);
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::cout << ") ---> ";
      RT::printArgs(R);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Close the function begin with a call to function end
    emitFunctionEndTrace(CorrelationID, PIFnName.c_str());
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

private:
  RT::PiPlugin MPlugin;
  const backend MBackend;
}; // class plugin
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
