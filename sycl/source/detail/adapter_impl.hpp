//==- adapter_impl.hpp ----------------------------------------------==//
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

#define __SYCL_UR_ERROR_REPORT(backend)                                        \
  std::string(sycl::detail::get_backend_name_no_vendor(backend)) +             \
      " backend failed with error: "

#define __SYCL_CHECK_UR_CODE_NO_EXC(expr, backend)                             \
  {                                                                            \
    auto code = expr;                                                          \
    if (code != UR_RESULT_SUCCESS) {                                           \
      std::cerr << __SYCL_UR_ERROR_REPORT(backend)                             \
                << sycl::detail::codeToString(code) << std::endl;              \
    }                                                                          \
  }

namespace sycl {
inline namespace _V1 {
enum class backend : char;
namespace detail {

/// The adapter class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
///
/// \ingroup sycl_ur
class adapter_impl {
public:
  adapter_impl() = delete;

  adapter_impl(ur_adapter_handle_t adapter, backend UseBackend)
      : MAdapter(adapter), MBackend(UseBackend),
        MAdapterMutex(std::make_shared<std::mutex>()) {

#ifdef _WIN32
    UrLoaderHandle = ur::getURLoaderLibrary();
    PopulateUrFuncPtrTable(&UrFuncPtrs, UrLoaderHandle);
#endif
  }

  // Disallow accidental copies of adapters
  adapter_impl &operator=(const adapter_impl &) = delete;
  adapter_impl(const adapter_impl &) = delete;
  adapter_impl &operator=(adapter_impl &&other) noexcept = delete;
  adapter_impl(adapter_impl &&other) noexcept = delete;

  ~adapter_impl() = default;

  /// \throw SYCL 2020 exception(errc) if ur_result is not UR_RESULT_SUCCESS
  template <sycl::errc errc = sycl::errc::runtime>
  void checkUrResult(ur_result_t ur_result) const {
    if (ur_result == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
      assert(!adapterReleased);
      const char *message = nullptr;
      int32_t adapter_error = 0;
      ur_result = call_nocheck<UrApiKind::urAdapterGetLastError>(
          MAdapter, &message, &adapter_error);
      throw sycl::detail::set_ur_error(
          sycl::exception(
              sycl::make_error_code(errc),
              __SYCL_UR_ERROR_REPORT(MBackend) +
                  sycl::detail::codeToString(ur_result) +
                  (message ? "\n" + std::string(message) + "(adapter error )" +
                                 std::to_string(adapter_error) + "\n"
                           : std::string{})),
          ur_result);
    }
    if (ur_result != UR_RESULT_SUCCESS) {
      throw sycl::detail::set_ur_error(
          sycl::exception(sycl::make_error_code(errc),
                          __SYCL_UR_ERROR_REPORT(MBackend) +
                              sycl::detail::codeToString(ur_result)),
          ur_result);
    }
  }

  std::vector<ur_platform_handle_t> &getUrPlatforms() {
    std::call_once(PlatformsPopulated, [&]() {
      uint32_t platformCount = 0;
      call<UrApiKind::urPlatformGet>(MAdapter, 0u, nullptr, &platformCount);
      UrPlatforms.resize(platformCount);
      if (platformCount) {
        call<UrApiKind::urPlatformGet>(MAdapter, platformCount,
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
  ur_result_t call_nocheck(ArgsT &&...Args) const {
    ur_result_t R = UR_RESULT_SUCCESS;
    if (!adapterReleased) {
      detail::UrFuncInfo<UrApiOffset> UrApiInfo;
      auto F = UrApiInfo.getFuncPtr(&UrFuncPtrs);
      R = F(std::forward<ArgsT>(Args)...);
    }
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<UrApiOffset>(std::forward<ArgsT>(Args)...);
    checkUrResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<UrApiOffset>(std::forward<ArgsT>(Args)...);
    checkUrResult<errc>(Err);
  }

  /// Returns the backend reported by the adapter.
  backend getBackend() const { return MBackend; }

  /// Tells if this adapter can serve specified backend.
  /// For example, Unified Runtime adapter will be able to serve
  /// multiple backends as determined by the platforms reported by the adapter.
  bool hasBackend(backend Backend) const { return Backend == MBackend; }

  void release() {
    auto Res = call_nocheck<UrApiKind::urAdapterRelease>(MAdapter);
    if (Res == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
      // We can't query the adapter for the error message because the adapter
      // has been released
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::runtime),
          __SYCL_UR_ERROR_REPORT(MBackend) +
              "Adapter failed to be released and reported "
              "`UR_RESULT_ERROR_ADAPTER_SPECIFIC`. This should "
              "never happen, please file a bug.");
    }
    this->adapterReleased = true;
    checkUrResult(Res);
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
}; // class adapter_impl

template <typename URResource> class Managed {
  static constexpr auto Release = []() constexpr {
    if constexpr (std::is_same_v<URResource, ur_program_handle_t>)
      return UrApiKind::urProgramRelease;
    if constexpr (std::is_same_v<URResource, ur_kernel_handle_t>)
      return UrApiKind::urKernelRelease;
  }();
  static constexpr auto Retain = []() constexpr {
    if constexpr (std::is_same_v<URResource, ur_program_handle_t>)
      return UrApiKind::urProgramRetain;
    if constexpr (std::is_same_v<URResource, ur_kernel_handle_t>)
      return UrApiKind::urKernelRetain;
  }();

public:
  Managed() = default;
  Managed(URResource R, adapter_impl &Adapter) : R(R), Adapter(&Adapter) {}
  Managed(adapter_impl &Adapter) : Adapter(&Adapter) {}
  Managed(const Managed &) = delete;
  Managed(Managed &&Other) : Adapter(Other.Adapter) {
    R = Other.R;
    Other.R = nullptr;
  }
  Managed &operator=(const Managed &) = delete;
  Managed &operator=(Managed &&Other) {
    URResource Temp = Other.R;
    Other.R = nullptr;
    if (R)
      Adapter->call<Release>(R);
    R = Temp;

    Adapter = Other.Adapter;
    return *this;
  }

  operator URResource() const { return R; }

  URResource release() {
    URResource Res = R;
    R = nullptr;
    return Res;
  }

  URResource *operator&() {
    assert(!R && "Already initialized!");
    assert(Adapter && "Adapter must be set for this API!");
    return &R;
  }

  ~Managed() {
    if (!R)
      return;

    Adapter->call<Release>(R);
  }

  Managed retain() {
    assert(R && "Cannot retain unintialized resource!");
    Adapter->call<Retain>(R);
    return Managed{R, *Adapter};
  }

  bool operator==(const Managed &Other) const {
    assert((!Adapter || !Other.Adapter || Adapter == Other.Adapter) &&
           "Objects must belong to the same adapter!");
    return R == Other.R;
  }

private:
  URResource R = nullptr;
  adapter_impl *Adapter = nullptr;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
