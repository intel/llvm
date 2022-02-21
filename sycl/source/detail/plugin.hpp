//==------------------------- plugin.hpp - SYCL platform -------------------==//
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
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/plugin_printers.hpp>
#include <memory>
#include <mutex>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GPICallEvent;
extern xpti::trace_event_data_t *GPIArgCallEvent;
#endif

template <PiApiKind Kind, size_t Idx, typename... Args>
struct array_fill_helper;

template <PiApiKind Kind> struct PiApiArgTuple;

#define _PI_API(api)                                                           \
  template <> struct PiApiArgTuple<PiApiKind::api> {                           \
    using type = typename function_traits<decltype(api)>::args_type;           \
  };

#include <CL/sycl/detail/pi.def>
#undef _PI_API

template <PiApiKind Kind, size_t Idx, typename T>
struct array_fill_helper<Kind, Idx, T> {
  static void fill(unsigned char *Dst, T &&Arg) {
    using ArgsTuple = typename PiApiArgTuple<Kind>::type;
    // C-style cast is required here.
    auto RealArg = (std::tuple_element_t<Idx, ArgsTuple>)(Arg);
    *(std::remove_cv_t<std::tuple_element_t<Idx, ArgsTuple>> *)Dst = RealArg;
  }
};

template <PiApiKind Kind, size_t Idx, typename T, typename... Args>
struct array_fill_helper<Kind, Idx, T, Args...> {
  static void fill(unsigned char *Dst, const T &&Arg, Args &&... Rest) {
    using ArgsTuple = typename PiApiArgTuple<Kind>::type;
    // C-style cast is required here.
    auto RealArg = (std::tuple_element_t<Idx, ArgsTuple>)(Arg);
    *(std::remove_cv_t<std::tuple_element_t<Idx, ArgsTuple>> *)Dst = RealArg;
    array_fill_helper<Kind, Idx + 1, Args...>::fill(
        Dst + sizeof(decltype(RealArg)), std::forward<Args>(Rest)...);
  }
};

template <typename... Ts>
constexpr size_t totalSize(const std::tuple<Ts...> &) {
  return (sizeof(Ts) + ...);
}

template <PiApiKind Kind, typename... ArgsT>
auto packCallArguments(ArgsT &&... Args) {
  using ArgsTuple = typename PiApiArgTuple<Kind>::type;

  constexpr size_t TotalSize = totalSize(ArgsTuple{});

  std::array<unsigned char, TotalSize> ArgsData;
  array_fill_helper<Kind, 0, ArgsT...>::fill(ArgsData.data(),
                                             std::forward<ArgsT>(Args)...);

  return ArgsData;
}

/// The plugin class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_pi
class plugin {
public:
  plugin() = delete;
  plugin(const std::shared_ptr<RT::PiPlugin> &Plugin, backend UseBackend,
         void *LibraryHandle)
      : MPlugin(Plugin), MBackend(UseBackend), MLibraryHandle(LibraryHandle),
        TracingMutex(std::make_shared<std::mutex>()),
        MPluginMutex(std::make_shared<std::mutex>()) {}

  plugin &operator=(const plugin &) = default;
  plugin(const plugin &) = default;
  plugin &operator=(plugin &&other) noexcept = default;
  plugin(plugin &&other) noexcept = default;

  ~plugin() = default;

  const RT::PiPlugin &getPiPlugin() const { return *MPlugin; }
  RT::PiPlugin &getPiPlugin() { return *MPlugin; }
  const std::shared_ptr<RT::PiPlugin> &getPiPluginPtr() const {
    return MPlugin;
  }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = cl::sycl::runtime_error>
  void checkPiResult(RT::PiResult pi_result) const {
    __SYCL_CHECK_OCL_CODE_THROW(pi_result, Exception);
  }

  /// \throw SYCL 2020 exception(errc) if pi_result is not PI_SUCCESS
  template <sycl::errc errc> void checkPiResult(RT::PiResult pi_result) const {
    __SYCL_CHECK_CODE_THROW_VIA_ERRC(pi_result, errc);
  }

  void reportPiError(RT::PiResult pi_result, const char *context) const {
    if (pi_result != PI_SUCCESS) {
      throw cl::sycl::runtime_error(
          std::string(context) + " API failed with error: " +
              cl::sycl::detail::codeToString(pi_result),
          pi_result);
    }
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
    const char *PIFnName = PiCallInfo.getFuncName();
    uint64_t CorrelationID = pi::emitFunctionBeginTrace(PIFnName);
    uint64_t CorrelationIDWithArgs = 0;
    unsigned char *ArgsDataPtr = nullptr;
    // TODO check if stream is observed when corresponding API is present.
    if (xptiTraceEnabled()) {
      auto ArgsData =
          packCallArguments<PiApiOffset>(std::forward<ArgsT>(Args)...);
      ArgsDataPtr = ArgsData.data();
      CorrelationIDWithArgs = pi::emitFunctionWithArgsBeginTrace(
          static_cast<uint32_t>(PiApiOffset), PIFnName, ArgsDataPtr, *MPlugin);
    }
#endif
    RT::PiResult R;
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      // MEmergencyModeEnabled signals if process is terminating now, it may
      // happen by initiative from underlying modules and mutex can be still
      // locked. Program behavior is undefined if thread terminates while owning
      // a mutex so skip mutex ownership request in emergency mode.
      auto Guard = MEmergencyModeEnabled
                       ? std::unique_lock<std::mutex>()
                       : std::unique_lock<std::mutex>(*TracingMutex);
      const char *FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      RT::printArgs(Args...);
      R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
      std::cout << ") ---> ";
      RT::printArgs(R);
      RT::printOuts(Args...);
      std::cout << std::endl;
    } else {
      R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Close the function begin with a call to function end
    pi::emitFunctionEndTrace(CorrelationID, PIFnName);
    pi::emitFunctionWithArgsEndTrace(CorrelationIDWithArgs,
                                     static_cast<uint32_t>(PiApiOffset),
                                     PIFnName, ArgsDataPtr, R, *MPlugin);
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

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    RT::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult<errc>(Err);
  }

  backend getBackend(void) const { return MBackend; }
  void *getLibraryHandle() const { return MLibraryHandle; }
  void *getLibraryHandle() { return MLibraryHandle; }
  int unload() { return RT::unloadPlugin(MLibraryHandle); }

  // return the index of PiPlatforms.
  // If not found, add it and return its index.
  // The function is expected to be called in a thread safe manner.
  int getPlatformId(RT::PiPlatform Platform) {
    auto It = std::find(PiPlatforms.begin(), PiPlatforms.end(), Platform);
    if (It != PiPlatforms.end())
      return It - PiPlatforms.begin();

    PiPlatforms.push_back(Platform);
    LastDeviceIds.push_back(0);
    return PiPlatforms.size() - 1;
  }

  // Device ids are consecutive across platforms within a plugin.
  // We need to return the same starting index for the given platform.
  // So, instead of returing the last device id of the given platform,
  // return the last device id of the predecessor platform.
  // The function is expected to be called in a thread safe manner.
  int getStartingDeviceId(RT::PiPlatform Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId == 0)
      return 0;
    return LastDeviceIds[PlatformId - 1];
  }

  // set the id of the last device for the given platform
  // The function is expected to be called in a thread safe manner.
  void setLastDeviceId(RT::PiPlatform Platform, int Id) {
    int PlatformId = getPlatformId(Platform);
    LastDeviceIds[PlatformId] = Id;
  }

  bool containsPiPlatform(RT::PiPlatform Platform) {
    auto It = std::find(PiPlatforms.begin(), PiPlatforms.end(), Platform);
    return It != PiPlatforms.end();
  }

  std::shared_ptr<std::mutex> getPluginMutex() { return MPluginMutex; }

  void enableEmergencyMode() { MEmergencyModeEnabled = true; }

private:
  std::shared_ptr<RT::PiPlugin> MPlugin;
  backend MBackend;
  void *MLibraryHandle; // the handle returned from dlopen
  std::shared_ptr<std::mutex> TracingMutex;
  // Mutex to guard PiPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MPluginMutex;
  // vector of PiPlatforms that belong to this plugin
  std::vector<RT::PiPlatform> PiPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in PiPlatforms vector.
  std::vector<int> LastDeviceIds;

  bool MEmergencyModeEnabled = false;
}; // class plugin
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
