//===--- msan_ze.cpp - Load the MSan Level Zero shim ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "msan_ze.hpp"
#include "sanitizer_log.hpp"

#include <dlfcn.h>
#include <string>

namespace llvm::offload::sanitizer {
namespace msan {

const MsanZeTable *ZeReal = nullptr;

bool loadZeShim() {
  Dl_info Info{};
  if (!dladdr(reinterpret_cast<void *>(&loadZeShim), &Info) || !Info.dli_fname)
    return false;
  std::string Path = Info.dli_fname;
  Path = Path.substr(0, Path.rfind('/') + 1) + "liboffload_msan_ze.so";

  // Never dlclosed: the plugin keeps binding to it as "libze_loader.so".
  void *Shim = dlopen(Path.c_str(), RTLD_NOW);
  offloadMsanZeSetHooks_t SetHooks = nullptr;
  if (Shim)
    SetHooks = reinterpret_cast<offloadMsanZeSetHooks_t>(
        dlsym(Shim, "offloadMsanZeSetHooks"));
  if (!SetHooks) {
    SANITIZER_LOG("MSan Level Zero shim not loaded ({}): {}", Path,
                  Shim ? "no offloadMsanZeSetHooks" : dlerror());
    return false;
  }
  ZeReal = SetHooks(getZeHooks());
  SANITIZER_LOG("MSan Level Zero shim loaded: {}", Path);
  // Same banner as the UR layer, which is not loaded in this mode.
  SANITIZER_REPORT("==== DeviceSanitizer: MSAN");
  return true;
}

} // namespace msan
} // namespace llvm::offload::sanitizer
