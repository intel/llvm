/*
 *
 * Copyright (C) 2026 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ze_shim.cpp
 *
 * libur_msan_ze.so: SONAME "libze_loader.so", DT_NEEDED "libze_loader.so.1".
 * Once loaded by the MSan layer, liboffload's dlopen("libze_loader.so") binds
 * here. Only the hooked calls are defined; everything else resolves through
 * DT_NEEDED to the real loader.
 */

#include "../msan_ze_hooks.hpp"

#include <atomic>
#include <dlfcn.h>

namespace {

std::atomic<const MsanZeTable *> Hooks{nullptr};

const MsanZeTable Real = [] {
  MsanZeTable T{};
  void *L = dlopen("libze_loader.so.1", RTLD_NOW);
#define MSAN_ZE_REAL(N, P, A)                                                  \
  T.N = L ? reinterpret_cast<decltype(T.N)>(dlsym(L, #N)) : nullptr;
  MSAN_ZE_HOOKS(MSAN_ZE_REAL)
#undef MSAN_ZE_REAL
  return T;
}();

} // namespace

#define MSAN_ZE_TRAMPOLINE(N, P, A)                                            \
  extern "C" ZE_APIEXPORT ze_result_t ZE_APICALL N P {                         \
    auto *H = Hooks.load();                                                    \
    return H ? H->N A : Real.N A;                                              \
  }
MSAN_ZE_HOOKS(MSAN_ZE_TRAMPOLINE)
#undef MSAN_ZE_TRAMPOLINE

// Forces liboffload's level_zero plugin onto the SetGroupSize +
// SetArgumentValue + AppendLaunchKernel path, which the hooks intercept.
extern "C" ZE_APIEXPORT ze_result_t ZE_APICALL
zeCommandListAppendLaunchKernelWithArguments(
    ze_command_list_handle_t, ze_kernel_handle_t, const ze_group_count_t,
    const ze_group_size_t, void **, const void *, ze_event_handle_t, uint32_t,
    ze_event_handle_t *) {
  return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

extern "C" ZE_APIEXPORT const MsanZeTable *
urMsanZeSetHooks(const MsanZeTable *H) {
  Hooks.store(H);
  return &Real;
}
