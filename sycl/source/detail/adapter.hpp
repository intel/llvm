//==- adapter.hpp ----------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/config.hpp>
#include <detail/ur.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/type_traits.hpp>

#include <ur_api.h>
#ifdef XPTI_ENABLE_INSTRUMENTATION
// Include the headers necessary for emitting traces using the trace framework
#include "xpti/xpti_trace_framework.h"
#endif

#include <memory>
#include <mutex>

#define __SYCL_CHECK_UR_CODE_NO_EXC(expr)                                      \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != UR_RESULT_SUCCESS) {                                           \
      std::cerr << __SYCL_UR_ERROR_REPORT << sycl::detail::codeToString(code)  \
                << std::endl;                                                  \
    }                                                                          \
  }

namespace sycl {
inline namespace _V1 {
namespace detail {

/// The adapter class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_ur
class Adapter {
public:
  Adapter() = delete;

  Adapter(ur_adapter_handle_t adapter, backend UseBackend)
      : MAdapter(adapter), MBackend(UseBackend),
        TracingMutex(std::make_shared<std::mutex>()),
        MAdapterMutex(std::make_shared<std::mutex>()) {

#ifdef _WIN32
    UrLoaderHandle = ur::getURLoaderLibrary();
    PopulateUrFuncPtrTable(&UrFuncPtrs, UrLoaderHandle);
#endif
  }

  // Disallow accidental copies of adapters
  Adapter &operator=(const Adapter &) = delete;
  Adapter(const Adapter &) = delete;
  Adapter &operator=(Adapter &&other) noexcept = delete;
  Adapter(Adapter &&other) noexcept = delete;

  ~Adapter() = default;

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
      if (platformCount) {
        call<UrApiKind::urPlatformGet>(&MAdapter, 1, platformCount,
                                       UrPlatforms.data(), nullptr);
      }
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
  /// ur_result_t Err = Adapter->call<UrApiKind::urEntryPoint>(Args);
  /// Adapter->checkUrResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa adapter::checkUrResult
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

  /// Tells if this adapter can serve specified backend.
  /// For example, Unified Runtime adapter will be able to serve
  /// multiple backends as determined by the platforms reported by the adapter.
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

  // Device ids are consecutive across platforms within a adapter.
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

  std::shared_ptr<std::mutex> getAdapterMutex() { return MAdapterMutex; }
  bool adapterReleased = false;

private:
  ur_adapter_handle_t MAdapter;
  backend MBackend;
  std::shared_ptr<std::mutex> TracingMutex;
  // Mutex to guard UrPlatforms and LastDeviceIds.
  // Note that this is a temporary solution until we implement the global
  // Device/Platform cache later.
  std::shared_ptr<std::mutex> MAdapterMutex;
  // vector of UrPlatforms that belong to this adapter
  std::once_flag PlatformsPopulated;
  std::vector<ur_platform_handle_t> UrPlatforms;
  // represents the unique ids of the last device of each platform
  // index of this vector corresponds to the index in UrPlatforms vector.
  std::vector<int> LastDeviceIds;
#ifdef _WIN32
  void *UrLoaderHandle = nullptr;
#endif
  UrFuncPtrMapT UrFuncPtrs;
}; // class Adapter

using AdapterPtr = std::shared_ptr<Adapter>;

} // namespace detail
} // namespace _V1
} // namespace sycl
