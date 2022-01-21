//==------------- PiMock.hpp --- Mock unit testing library -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This mini-library provides facilities to test the DPC++ Runtime behavior upon
// specific results of the underlying low-level API calls. By exploiting the
// Plugin Interface API, the stored addresses of the actual plugin-specific
// implementations can be overwritten to point at user-defined mock functions.
//
// While this could be done manually for each unit-testing scenario, the library
// aims to rule out the boilerplate, providing helper APIs which can be re-used
// by all such unit tests. The test code stemming from this can be more consise,
// with little difference from non-mock classes' usage.
//
// The following unit testing scenarios are thereby simplified:
// 1) testing the DPC++ RT management of specific PI return codes;
// 2) coverage of corner-cases related to specific data outputs
//    from underlying runtimes;
// 3) testing the order of PI API calls;
// ..., etc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/queue.hpp>
#include <detail/platform_impl.hpp>

#include <functional>
#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace unittest {

namespace detail = cl::sycl::detail;
namespace RT = detail::pi;

template <typename T> struct DispatchHelper;

template <typename R, typename... Args> struct DispatchHelper<R(Args...)> {
  using type = std::function<R(Args...)>;
};

struct PiDispatch {
#define _PI_API(api)                                                           \
  typename DispatchHelper<decltype(api)>::type mock_##api = [](auto...) {      \
    std::cerr << "Unexpected PI call\n";                                       \
    std::terminate();                                                          \
    return PI_SUCCESS;                                                         \
  };

#include <CL/sycl/detail/pi.def>

#undef _PI_API
};

extern std::unordered_map<backend, PiDispatch> GDispatchTables;

void hijackPlugins();

/// Overwites PI call for a single backend.
///
/// \param F is any callable (function or lambda), that will be used instead of
/// default PI call handler.
template <detail::PiApiKind PiKind, backend Backend, typename Ret,
          typename... Args>
void redefineOne(const std::function<Ret(Args...)> &F) {
#define _PI_API(api)                                                           \
  if constexpr (PiKind == detail::PiApiKind::api) {                            \
    GDispatchTables[Backend].mock_##api = F;                                   \
  }
#include <CL/sycl/detail/pi.def>

#undef _PI_API
}

template <detail::PiApiKind PiKind, backend Backend, typename Ret,
          typename... Args>
void redefineOne(Ret (*FP)(Args...)) {
  redefineOne<PiKind, Backend>(std::function{FP});
}

/// Overwites PI call for all backends.
///
/// \param F is any callable (function or lambda), that will be used instead of
/// default PI call handler.
template <detail::PiApiKind PiKind, typename Ret, typename... Args>
void redefine(const std::function<Ret(Args...)> &F) {
  redefineOne<PiKind, backend::opencl>(F);
  redefineOne<PiKind, backend::ext_oneapi_level_zero>(F);
  redefineOne<PiKind, backend::ext_oneapi_cuda>(F);
  redefineOne<PiKind, backend::ext_oneapi_hip>(F);
  redefineOne<PiKind, backend::ext_intel_esimd_emulator>(F);
}

template <detail::PiApiKind PiKind, typename Ret, typename... Args>
void redefine(Ret (*FP)(Args...)) {
  redefine<PiKind>(std::function{FP});
}

void setupDefaultMockAPIs();
void resetMockAPIs();

} // namespace unittest
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
