//===--- msan_ze_shim.cpp - libze_loader stand-in for MSan ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// liboffload_msan_ze.so: SONAME "libze_loader.so", DT_NEEDED
// "libze_loader.so.1". Once loaded by loadZeShim(), the plugin's
// dlopen("libze_loader.so") binds here. Only the hooked calls are defined;
// everything else resolves through DT_NEEDED to the real loader.
//
//===----------------------------------------------------------------------===//

#include "../msan_ze_hooks.hpp"

#include <atomic>
#include <dlfcn.h>

namespace {

std::atomic<const MsanZeTable *> Hooks{nullptr};

const MsanZeTable &getReal() {
  static const MsanZeTable Real = [] {
    MsanZeTable T{};
    void *L = dlopen("libze_loader.so.1", RTLD_NOW);
#define MSAN_ZE_REAL(N, P, A)                                                  \
  T.N = L ? reinterpret_cast<decltype(T.N)>(dlsym(L, #N)) : nullptr;
    MSAN_ZE_HOOKS(MSAN_ZE_REAL)
#undef MSAN_ZE_REAL
    return T;
  }();
  return Real;
}

} // namespace

#define MSAN_ZE_TRAMPOLINE(N, P, A)                                            \
  extern "C" ZE_APIEXPORT ze_result_t ZE_APICALL N P {                         \
    auto *H = Hooks.load();                                                    \
    return H ? H->N A : getReal().N A;                                         \
  }
MSAN_ZE_HOOKS(MSAN_ZE_TRAMPOLINE)
#undef MSAN_ZE_TRAMPOLINE

// Forces the level_zero plugin onto the SetGroupSize + SetArgumentValue +
// AppendLaunchKernel path, which the hooks intercept.
extern "C" ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernelWithArguments(ze_command_list_handle_t,
                                             ze_kernel_handle_t,
                                             const ze_group_count_t,
                                             const ze_group_size_t, void **,
                                             const void *, ze_event_handle_t,
                                             uint32_t, ze_event_handle_t *) {
  return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

extern "C" ZE_APIEXPORT const MsanZeTable *
offloadMsanZeSetHooks(const MsanZeTable *H) {
  Hooks.store(H);
  return &getReal();
}
