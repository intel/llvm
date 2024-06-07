//==------------------------- plugin.hpp - SYCL platform -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/config.hpp>
#include <detail/plugin_printers.hpp>
#include <memory>
#include <mutex>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/type_traits.hpp>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

#include <sycl/detail/iostream_proxy.hpp>

#define __SYCL_REPORT_PI_ERR_TO_STREAM(expr)                                   \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      std::cerr << __SYCL_PI_ERROR_REPORT << sycl::detail::codeToString(code)  \
                << std::endl;                                                  \
    }                                                                          \
  }

#define __SYCL_REPORT_PI_ERR_TO_EXC(expr, exc, str)                            \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      std::string err_str =                                                    \
          str ? "\n" + std::string(str) + "\n" : std::string{};                \
      throw exc(__SYCL_PI_ERROR_REPORT + sycl::detail::codeToString(code) +    \
                    err_str,                                                   \
                code);                                                         \
    }                                                                          \
  }

#define __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(expr, errc)                          \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != PI_SUCCESS) {                                                  \
      throw sycl::exception(sycl::make_error_code(errc),                       \
                            __SYCL_PI_ERROR_REPORT +                           \
                                sycl::detail::codeToString(code));             \
    }                                                                          \
  }

#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC, STR)                               \
  __SYCL_REPORT_PI_ERR_TO_EXC(X, EXC, STR)
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) __SYCL_REPORT_PI_ERR_TO_STREAM(X)

#define __SYCL_CHECK_CODE_THROW_VIA_ERRC(X, ERRC)                              \
  __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(X, ERRC)

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
extern xpti::trace_event_data_t *GPICallEvent;
extern xpti::trace_event_data_t *GPIArgCallEvent;
extern uint8_t PiCallStreamID;
extern uint8_t PiDebugCallStreamID;
#endif

template <PiApiKind Kind, size_t Idx, typename... Args>
struct array_fill_helper;

template <PiApiKind Kind> struct PiApiArgTuple;

#define _PI_API(api)                                                           \
  template <> struct PiApiArgTuple<PiApiKind::api> {                           \
    using type = typename function_traits<decltype(api)>::args_type;           \
  };

#include <sycl/detail/pi.def>
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
  static void fill(unsigned char *Dst, const T &&Arg, Args &&...Rest) {
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
auto packCallArguments(ArgsT &&...Args) {
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
  plugin(const std::shared_ptr<sycl::detail::pi::PiPlugin> &Plugin,
         backend UseBackend, void *LibraryHandle)
      : MPlugin(Plugin), MBackend(UseBackend), MLibraryHandle(LibraryHandle),
        TracingMutex(std::make_shared<std::mutex>()),
        MPluginMutex(std::make_shared<std::mutex>()) {}

  // Disallow accidental copies of plugins
  plugin &operator=(const plugin &) = delete;
  plugin(const plugin &) = delete;
  plugin &operator=(plugin &&other) noexcept = delete;
  plugin(plugin &&other) noexcept = delete;

  ~plugin() = default;

  const sycl::detail::pi::PiPlugin &getPiPlugin() const { return *MPlugin; }
  sycl::detail::pi::PiPlugin &getPiPlugin() { return *MPlugin; }
  const std::shared_ptr<sycl::detail::pi::PiPlugin> &getPiPluginPtr() const {
    return MPlugin;
  }

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = sycl::runtime_error>
  void checkPiResult(sycl::detail::pi::PiResult pi_result) const {
    char *message = nullptr;
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }
    __SYCL_CHECK_OCL_CODE_THROW(pi_result, Exception, message);
  }

  /// \throw SYCL 2020 exception(errc) if pi_result is not PI_SUCCESS
  template <sycl::errc errc>
  void checkPiResult(sycl::detail::pi::PiResult pi_result) const {
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      char *message = nullptr;
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }
    __SYCL_CHECK_CODE_THROW_VIA_ERRC(pi_result, errc);
  }

  void reportPiError(sycl::detail::pi::PiResult pi_result,
                     const char *context) const {
    if (pi_result != PI_SUCCESS) {
      throw sycl::runtime_error(std::string(context) +
                                    " API failed with error: " +
                                    sycl::detail::codeToString(pi_result),
                                pi_result);
    }
  }

  /// Calls the PiApi, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// PiResult Err = Plugin->call<PiApiKind::pi>(Args);
  /// Plugin->checkPiResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa plugin::checkPiResult

  template <PiApiKind PiApiOffset, typename... ArgsT>
  sycl::detail::pi::PiResult call_nocheck(ArgsT... Args) const {
    sycl::detail::pi::PiFuncInfo<PiApiOffset> PiCallInfo;
#ifdef XPTI_ENABLE_INSTRUMENTATION
    bool CorrelationIDAvailable = false, CorrelationIDWithArgsAvailable = false;
    // Emit a function_begin trace for the PI API before the call is executed.
    // If arguments need to be captured, then a data structure can be sent in
    // the per_instance_user_data field.
    const char *PIFnName = PiCallInfo.getFuncName();
    uint64_t CorrelationIDWithArgs = 0, CorrelationID = 0;

    if (xptiCheckTraceEnabled(
            PiCallStreamID,
            (uint16_t)xpti::trace_point_type_t::function_begin)) {
      CorrelationID = pi::emitFunctionBeginTrace(PIFnName);
      CorrelationIDAvailable = true;
    }
    using PackCallArgumentsTy =
        decltype(packCallArguments<PiApiOffset>(std::forward<ArgsT>(Args)...));
    std::unique_ptr<PackCallArgumentsTy> ArgsDataPtr = nullptr;
    // If subscribers are listening to Pi debug call stream, only then prepare
    // the data for the notifications and emit notifications. Even though the
    // function emitFunctionWithArgsBeginTrace() checks for the trqace typoe
    // using xptiTraceCheckEnabled(), we add a guard here before we prepare the
    // data for the notification, as it comes with a cost
    if (xptiCheckTraceEnabled(
            PiDebugCallStreamID,
            (uint16_t)xpti::trace_point_type_t::function_with_args_begin)) {
      // TODO check if stream is observed when corresponding API is present.
      ArgsDataPtr = std::make_unique<PackCallArgumentsTy>(
          xptiTraceEnabled()
              ? packCallArguments<PiApiOffset>(std::forward<ArgsT>(Args)...)
              : PackCallArgumentsTy{});
      CorrelationIDWithArgs = pi::emitFunctionWithArgsBeginTrace(
          static_cast<uint32_t>(PiApiOffset), PIFnName, ArgsDataPtr->data(),
          *MPlugin);
      CorrelationIDWithArgsAvailable = true;
    }
#endif
    sycl::detail::pi::PiResult R = PI_SUCCESS;
    if (pi::trace(pi::TraceLevel::PI_TRACE_CALLS)) {
      std::lock_guard<std::mutex> Guard(*TracingMutex);
      const char *FnName = PiCallInfo.getFuncName();
      std::cout << "---> " << FnName << "(" << std::endl;
      sycl::detail::pi::printArgs(Args...);
      if (!pluginReleased) {
        R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
        std::cout << ") ---> ";
        sycl::detail::pi::printArgs(R);
        sycl::detail::pi::printOuts(Args...);
        std::cout << std::endl;
      } else {
        std::cout << ") ---> ";
        std::cout << "API Called After Plugin Teardown, Functon Call ignored.";
        std::cout << std::endl;
      }
    } else {
      if (!pluginReleased) {
        R = PiCallInfo.getFuncPtr(*MPlugin)(Args...);
      }
    }
#ifdef XPTI_ENABLE_INSTRUMENTATION
    // Close the function begin with a call to function end; we do not need to
    // check th xptiTraceCheckEnbled() here as it is performed within the
    // function
    if (CorrelationIDAvailable) {
      // Only send function_end notification if function_begin is subscribed to
      pi::emitFunctionEndTrace(CorrelationID, PIFnName);
    }
    if (CorrelationIDWithArgsAvailable) {
      pi::emitFunctionWithArgsEndTrace(
          CorrelationIDWithArgs, static_cast<uint32_t>(PiApiOffset), PIFnName,
          ArgsDataPtr->data(), R, *MPlugin);
    }
#endif
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    sycl::detail::pi::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, PiApiKind PiApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    sycl::detail::pi::PiResult Err = call_nocheck<PiApiOffset>(Args...);
    checkPiResult<errc>(Err);
  }

  /// Tells if this plugin can serve specified backend.
  /// For example, Unified Runtime plugin will be able to serve
  /// multiple backends as determined by the platforms reported by the plugin.
  bool hasBackend(backend Backend) const { return Backend == MBackend; }

  void *getLibraryHandle() const { return MLibraryHandle; }
  void *getLibraryHandle() { return MLibraryHandle; }
  int unload() {
    this->pluginReleased = true;
    return sycl::detail::pi::unloadPlugin(MLibraryHandle);
  }

  // return the index of PiPlatforms.
  // If not found, add it and return its index.
  // The function is expected to be called in a thread safe manner.
  int getPlatformId(sycl::detail::pi::PiPlatform Platform) {
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
  int getStartingDeviceId(sycl::detail::pi::PiPlatform Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId == 0)
      return 0;
    return LastDeviceIds[PlatformId - 1];
  }

  // set the id of the last device for the given platform
  // The function is expected to be called in a thread safe manner.
  void setLastDeviceId(sycl::detail::pi::PiPlatform Platform, int Id) {
    int PlatformId = getPlatformId(Platform);
    LastDeviceIds[PlatformId] = Id;
  }

  // Adjust the id of the last device for the given platform.
  // Involved when there is no device on that platform at all.
  // The function is expected to be called in a thread safe manner.
  void adjustLastDeviceId(sycl::detail::pi::PiPlatform Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId > 0 &&
        LastDeviceIds[PlatformId] < LastDeviceIds[PlatformId - 1])
      LastDeviceIds[PlatformId] = LastDeviceIds[PlatformId - 1];
  }

  bool containsPiPlatform(sycl::detail::pi::PiPlatform Platform) {
    auto It = std::find(PiPlatforms.begin(), PiPlatforms.end(), Platform);
    return It != PiPlatforms.end();
  }

  std::shared_ptr<std::mutex> getPluginMutex() { return MPluginMutex; }
  bool pluginReleased = false;

private:
  std::shared_ptr<sycl::detail::pi::PiPlugin> MPlugin;
  backend MBackend;
  void *MLibraryHandle; // the handle returned from dlopen
  std::shared_ptr<std::mutex> TracingMutex;
  // Mutex to guard PiPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MPluginMutex;
  // vector of PiPlatforms that belong to this plugin
  std::vector<sycl::detail::pi::PiPlatform> PiPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in PiPlatforms vector.
  std::vector<int> LastDeviceIds;
}; // class plugin

using PluginPtr = std::shared_ptr<plugin>;

} // namespace detail
} // namespace _V1
} // namespace sycl
