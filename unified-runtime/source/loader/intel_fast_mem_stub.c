/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file intel_fast_mem_stub.c
 *
 */

// Stub implementations for Intel fast memory functions.
//
// The pre-built LLVM static archives that the symbolizer pulls in (LLVMSupport,
// LLVMSymbolize, LLVMDebugInfoDWARF/GSYM/PDB, LLVMObject, LLVMDemangle, ...)
// are compiled by icx, which by default emits calls to
// _intel_fast_memcpy/_intel_fast_memset. Those symbols are normally provided by
// libirc. ur_loader (and libsycl, which archives it) doesn't link with libirc,
// so forward these calls to plain memcpy/memset instead.

#include <string.h>

void *_intel_fast_memcpy(void *dest, const void *src, size_t n) {
  return memcpy(dest, src, n);
}

void *_intel_fast_memset(void *dest, int c, size_t n) {
  return memset(dest, c, n);
}
