//==------------------------- plugin.hpp - SYCL platform -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/config.hpp>
#include <memory>
#include <mutex>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/ur.hpp>

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

#define __SYCL_CHECK_OCL_CODE_NO_EXC(X) __SYCL_REPORT_UR_ERR_TO_STREAM(X)

namespace sycl {
inline namespace _V1 {
namespace detail {

/// The plugin class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_ur
class plugin {
public:
  plugin() = delete;

  plugin(ur_adapter_handle_t adapter, backend UseBackend)
      : MAdapter(adapter), MBackend(UseBackend),
        TracingMutex(std::make_shared<std::mutex>()),
        MPluginMutex(std::make_shared<std::mutex>()) {

#ifdef _WIN32
    UrLoaderHandle = ur::getURLoaderLibrary();
    PopulateUrFuncPtrTable(&UrFuncPtrs, UrLoaderHandle);
#endif
  }

  // Disallow accidental copies of plugins
  plugin &operator=(const plugin &) = delete;
  plugin(const plugin &) = delete;
  plugin &operator=(plugin &&other) noexcept = delete;
  plugin(plugin &&other) noexcept = delete;

  ~plugin() = default;

  /// \throw SYCL 2020 exception(errc) if ur_result is not UR_RESULT_SUCCESS
  template <sycl::errc errc = sycl::errc::runtime>
  void checkUrResult(ur_result_t ur_result) const {
    const char *message = nullptr;
    if (ur_result == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
      int32_t adapter_error = 0;
      ur_result = call_nocheck<UrApiKind::urAdapterGetLastError>(
          MAdapter, &message, &adapter_error);

      // If the warning level is greater then 2 emit the message
      if (message != nullptr &&
          detail::SYCLConfig<detail::SYCL_RT_WARNING_LEVEL>::get() >= 2) {
        std::clog << message << std::endl;
      }

      // If it is a warning do not throw code
      if (ur_result == UR_RESULT_SUCCESS) {
        return;
      }
    }
    if (ur_result != UR_RESULT_SUCCESS) {
      throw sycl::detail::set_ur_error(
          sycl::exception(sycl::make_error_code(errc),
                          __SYCL_UR_ERROR_REPORT +
                              sycl::detail::codeToString(ur_result) +
                              (message ? "\n" + std::string(message) + "\n"
                                       : std::string{})),
          ur_result);
    }
  }

  std::vector<ur_platform_handle_t> &getUrPlatforms() {
    std::call_once(PlatformsPopulated, [&]() {
      uint32_t platformCount = 0;
      call<UrApiKind::urPlatformGet>(&MAdapter, 1, 0, nullptr, &platformCount);
      UrPlatforms.resize(platformCount);
      call<UrApiKind::urPlatformGet>(&MAdapter, 1, platformCount,
                                     UrPlatforms.data(), nullptr);
      // We need one entry in this per platform
      LastDeviceIds.resize(platformCount);
    });
    return UrPlatforms;
  }

  ur_adapter_handle_t getUrAdapter() { return MAdapter; }

  /// Calls the UR Api, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// ur_result_t Err = Plugin->call<UrApiKind::urEntryPoint>(Args);
  /// Plugin->checkUrResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa plugin::checkUrResult
  template <UrApiKind UrApiOffset, typename... ArgsT>
  ur_result_t call_nocheck(ArgsT... Args) const {
    ur_result_t R = UR_RESULT_SUCCESS;
    if (!adapterReleased) {
      detail::UrFuncInfo<UrApiOffset> UrApiInfo;
      auto F = UrApiInfo.getFuncPtr(&UrFuncPtrs);
      R = F(Args...);
    }
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    auto Err = call_nocheck<UrApiOffset>(Args...);
    checkUrResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT... Args) const {
    auto Err = call_nocheck<UrApiOffset>(Args...);
    checkUrResult<errc>(Err);
  }

  /// Tells if this plugin can serve specified backend.
  /// For example, Unified Runtime plugin will be able to serve
  /// multiple backends as determined by the platforms reported by the plugin.
  bool hasBackend(backend Backend) const { return Backend == MBackend; }

  void release() {
    call<UrApiKind::urAdapterRelease>(MAdapter);
    this->adapterReleased = true;
  }

  // Return the index of a UR platform.
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
  // Mutex to guard UrPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MPluginMutex;
  // vector of UrPlatforms that belong to this plugin
  std::once_flag PlatformsPopulated;
  std::vector<ur_platform_handle_t> UrPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in UrPlatforms vector.
  std::vector<int> LastDeviceIds;
#ifdef _WIN32
  void *UrLoaderHandle = nullptr;
#endif
  UrFuncPtrMapT UrFuncPtrs;
}; // class plugin

using PluginPtr = std::shared_ptr<plugin>;

} // namespace detail
} // namespace _V1
} // namespace sycl
