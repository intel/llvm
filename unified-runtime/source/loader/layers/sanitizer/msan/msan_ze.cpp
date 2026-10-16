/*
 *
 * Copyright (C) 2026 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ze.cpp
 *
 */

#include "msan_ze.hpp"
#include "ur_sanitizer_layer.hpp"

#include <dlfcn.h>
#include <string>

namespace ur_sanitizer_layer {
namespace msan {

const MsanZeTable *ZeReal = nullptr;

ur_result_t loadZeShim() {
  Dl_info Info{};
  if (!dladdr(reinterpret_cast<void *>(&loadZeShim), &Info) ||
      !Info.dli_fname)
    return UR_RESULT_SUCCESS;
  std::string Path = Info.dli_fname;
  Path = Path.substr(0, Path.rfind('/') + 1) + "libur_msan_ze.so";

  // Never dlclosed: liboffload keeps binding to it as "libze_loader.so".
  void *Shim = dlopen(Path.c_str(), RTLD_NOW);
  urMsanZeSetHooks_t SetHooks = nullptr;
  if (Shim)
    SetHooks = reinterpret_cast<urMsanZeSetHooks_t>(
        dlsym(Shim, "urMsanZeSetHooks"));
  if (!SetHooks) {
    UR_LOG_L(getContext()->logger, INFO,
             "MSan Level Zero shim not loaded ({}): {}", Path,
             Shim ? "no urMsanZeSetHooks" : dlerror());
    return UR_RESULT_SUCCESS;
  }
  ZeReal = SetHooks(getZeHooks());
  UR_LOG_L(getContext()->logger, INFO, "MSan Level Zero shim loaded: {}",
           Path);
  return UR_RESULT_SUCCESS;
}

} // namespace msan
} // namespace ur_sanitizer_layer
