/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file symbolizer_loader.cpp
 *
 */

#include "sanitizer_common/sanitizer_stacktrace.hpp"

#include <dlfcn.h>

extern "C" {

SymbolizeCodeFunction *TryLoadingSymbolizer() {
  SymbolizeCodeFunction *SymbolizeCode = nullptr;
  void *Handle = dlopen("libur_sanitizer_symbolizer.so", RTLD_LAZY);
  if (Handle) {
    SymbolizeCode = (SymbolizeCodeFunction *)dlsym(Handle, "SymbolizeCode");
    if (!SymbolizeCode) {
      dlclose(Handle);
    }
  }
  return SymbolizeCode;
}

} // extern "C"
