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

#include <ur_api.h>

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

#include <sycl/detail/iostream_proxy.hpp>

#define __SYCL_REPORT_UR_ERR_TO_STREAM(expr)                                   \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != UR_RESULT_SUCCESS) {                                           \
      std::cerr << __SYCL_UR_ERROR_REPORT << sycl::detail::codeToString(code)  \
                << std::endl;                                                  \
    }                                                                          \
  }

#define __SYCL_REPORT_UR_ERR_TO_EXC(expr, exc, str)                            \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != UR_RESULT_SUCCESS) {                                           \
      std::string err_str =                                                    \
          str ? "\n" + std::string(str) + "\n" : std::string{};                \
      throw exc(__SYCL_UR_ERROR_REPORT + sycl::detail::codeToString(code) +    \
                    err_str,                                                   \
                code);                                                         \
    }                                                                          \
  }

#define __SYCL_REPORT_ERR_TO_EXC_VIA_ERRC(expr, errc)                          \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != UR_RESULT_SUCCESS) {                                           \
      throw sycl::exception(sycl::make_error_code(errc),                       \
                            __SYCL_UR_ERROR_REPORT +                           \
                                sycl::detail::codeToString(code));             \
    }                                                                          \
  }

#define __SYCL_CHECK_OCL_CODE_THROW(X, EXC, STR)                               \
  __SYCL_REPORT_UR_ERR_TO_EXC(X, EXC, STR)
#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) __SYCL_REPORT_UR_ERR_TO_STREAM(X)

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
class urPlugin {
public:
  urPlugin() = delete;

  urPlugin(ur_adapter_handle_t adapter, backend UseBackend)
      : MAdapter(adapter), MBackend(UseBackend),
        TracingMutex(std::make_shared<std::mutex>()),
        MPluginMutex(std::make_shared<std::mutex>()) {}

  // Disallow accidental copies of plugins
  urPlugin &operator=(const urPlugin &) = delete;
  urPlugin(const urPlugin &) = delete;
  urPlugin &operator=(urPlugin &&other) noexcept = delete;
  urPlugin(urPlugin &&other) noexcept = delete;

  ~urPlugin() = default;

  /// Checks return value from PI calls.
  ///
  /// \throw Exception if pi_result is not a PI_SUCCESS.
  template <typename Exception = sycl::runtime_error>
  void checkUrResult(ur_result_t result) const {
    char *message = nullptr;
    /* TODO: hook up adapter specific error
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }*/
    __SYCL_CHECK_OCL_CODE_THROW(result, Exception, message);
  }

  /// \throw SYCL 2020 exception(errc) if pi_result is not PI_SUCCESS
  template <sycl::errc errc> void checkUrResult(ur_result_t result) const {
    /*
    if (pi_result == PI_ERROR_PLUGIN_SPECIFIC_ERROR) {
      char *message = nullptr;
      pi_result = call_nocheck<PiApiKind::piPluginGetLastError>(&message);

      // If the warning level is greater then 2 emit the message
      if (detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2)
        std::clog << message << std::endl;

      // If it is a warning do not throw code
      if (pi_result == PI_SUCCESS)
        return;
    }*/
    __SYCL_CHECK_CODE_THROW_VIA_ERRC(result, errc);
  }

  void reportUrError(ur_result_t ur_result, const char *context) const {
    if (ur_result != UR_RESULT_SUCCESS) {
      throw sycl::runtime_error(std::string(context) +
                                    " API failed with error: " +
                                    sycl::detail::codeToString(ur_result),
                                ur_result);
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

  std::vector<ur_platform_handle_t> &getUrPlatforms() {
    std::call_once(PlatformsPopulated, [&]() {
      uint32_t platformCount = 0;
      call(urPlatformGet, &MAdapter, 1, 0, nullptr, &platformCount);
      UrPlatforms.resize(platformCount);
      call(urPlatformGet, &MAdapter, 1, platformCount, UrPlatforms.data(),
           nullptr);
      // We need one entry in this per platform
      LastDeviceIds.resize(platformCount);
    });
    return UrPlatforms;
  }

  template <class UrFunc, typename... ArgsT>
  ur_result_t call_nocheck(UrFunc F, ArgsT... Args) const {
    ur_result_t R = UR_RESULT_SUCCESS;
    if (!adapterReleased) {
      R = F(Args...);
    }
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <class UrFunc, typename... ArgsT>
  void call(UrFunc F, ArgsT... Args) const {
    auto Err = call_nocheck(F, Args...);
    checkUrResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, class UrFunc, typename... ArgsT>
  void call(UrFunc F, ArgsT... Args) const {
    auto Err = call_nocheck(F, Args...);
    checkUrResult<errc>(Err);
  }

  /// Tells if this plugin can serve specified backend.
  /// For example, Unified Runtime plugin will be able to serve
  /// multiple backends as determined by the platforms reported by the plugin.
  bool hasBackend(backend Backend) const { return Backend == MBackend; }

  void release() {
    call(urAdapterRelease, MAdapter);
    this->adapterReleased = true;
    // This is where urAdapterRelease happens - only gets called in sycl RT
    // right next to piTeardown
    // return sycl::detail::pi::unloadPlugin(MLibraryHandle);
  }

  // return the index of PiPlatforms.
  // If not found, add it and return its index.
  // The function is expected to be called in a thread safe manner.
  int getPlatformId(ur_platform_handle_t Platform) {
    auto It = std::find(UrPlatforms.begin(), UrPlatforms.end(), Platform);
    if (It != UrPlatforms.end())
      return It - UrPlatforms.begin();

    UrPlatforms.push_back(Platform);
    LastDeviceIds.push_back(0);
    return UrPlatforms.size() - 1;
  }

  // Device ids are consecutive across platforms within a plugin.
  // We need to return the same starting index for the given platform.
  // So, instead of returing the last device id of the given platform,
  // return the last device id of the predecessor platform.
  // The function is expected to be called in a thread safe manner.
  int getStartingDeviceId(ur_platform_handle_t Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId == 0)
      return 0;
    return LastDeviceIds[PlatformId - 1];
  }

  // set the id of the last device for the given platform
  // The function is expected to be called in a thread safe manner.
  void setLastDeviceId(ur_platform_handle_t Platform, int Id) {
    int PlatformId = getPlatformId(Platform);
    LastDeviceIds[PlatformId] = Id;
  }

  // Adjust the id of the last device for the given platform.
  // Involved when there is no device on that platform at all.
  // The function is expected to be called in a thread safe manner.
  void adjustLastDeviceId(ur_platform_handle_t Platform) {
    int PlatformId = getPlatformId(Platform);
    if (PlatformId > 0 &&
        LastDeviceIds[PlatformId] < LastDeviceIds[PlatformId - 1])
      LastDeviceIds[PlatformId] = LastDeviceIds[PlatformId - 1];
  }

  bool containsUrPlatform(ur_platform_handle_t Platform) {
    auto It = std::find(UrPlatforms.begin(), UrPlatforms.end(), Platform);
    return It != UrPlatforms.end();
  }

  std::shared_ptr<std::mutex> getPluginMutex() { return MPluginMutex; }
  bool adapterReleased = false;

private:
  ur_adapter_handle_t MAdapter;
  backend MBackend;
  std::shared_ptr<std::mutex> TracingMutex;
  // Mutex to guard PiPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MPluginMutex;
  // vector of PiPlatforms that belong to this plugin
  std::once_flag PlatformsPopulated;
  std::vector<ur_platform_handle_t> UrPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in PiPlatforms vector.
  std::vector<int> LastDeviceIds;
}; // class plugin

using UrPluginPtr = std::shared_ptr<urPlugin>;

} // namespace detail
} // namespace _V1
} // namespace sycl
