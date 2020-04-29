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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

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

/// Two plugins are the same if their string is the same.
/// There is no need to check the actual string, just the pointer, since
/// there is only one instance of the PiPlugin struct per backend.
///
/// \ingroup sycl_pi
///
inline bool operator==(const plugin &lhs, const plugin &rhs) {
  return (lhs.getBackend() == rhs.getBackend());
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
