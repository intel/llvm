//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/kernel_desc.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace unittest {
struct MockKernelInfoBase {
  static constexpr detail::kernel_param_desc_t Dummy{};

  static constexpr unsigned getNumParams() { return 0; }
  static constexpr const detail::kernel_param_desc_t &getParamDesc(int) {
    return Dummy;
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }

  static constexpr const char *getFileName() { return ""; }
  static constexpr const char *getFunctionName() { return ""; }
  static constexpr unsigned getLineNumber() { return 0; }
  static constexpr unsigned getColumnNumber() { return 0; }
};


// Instantiating this function registers kernel names in KernelRegistry,
// as would normally be done by a kernel wrapper at its submisison.
template <typename... KernelNames> void registerKernelNames() {
  (sycl::detail::KernelRegistrar<
       KernelNames,
       sycl::detail::KernelInfo<KernelNames>>::registerKernelName(),
   ...);
}
} // namespace unittest
} // namespace _V1
} // namespace sycl

// In most cases we don't need to redefine any other method besides getName(),
// so here we only have the simplest helper. If any test needs to redefine more
// methods, they can do that explicitly.
// While kernel name registry is not part of the integration header, all kernels
// that have one need to have the other, so add the instantiation here.
#define MOCK_INTEGRATION_HEADER(KernelName)                                    \
  namespace sycl {                                                             \
  inline namespace _V1 {                                                       \
  namespace detail {                                                           \
  template <>                                                                  \
  struct KernelInfo<KernelName> : public unittest::MockKernelInfoBase {        \
    static constexpr const char *getName() { return #KernelName; }             \
  };                                                                           \
  } /* namespace detail */                                                     \
  namespace unittest {                                                         \
  template void registerKernelNames<KernelName>();                             \
  } /* namespace unittest */                                                   \
  } /* namespace _V1 */                                                        \
  } /* namespace sycl */
