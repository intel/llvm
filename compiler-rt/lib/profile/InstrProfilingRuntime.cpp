//===- InstrProfilingRuntime.cpp - PGO runtime initialization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" {

#include "InstrProfiling.h"

void __sycl_increment_profile_counters(uint64_t FnHash, size_t NumCounters,
                                       const uint64_t *Increments) {
  for (const __llvm_profile_data *DataVar = __llvm_profile_begin_data();
       DataVar < __llvm_profile_end_data(); DataVar++) {
    if (DataVar->NameRef != FnHash || DataVar->NumCounters != NumCounters)
      continue;

    uint64_t *const Counters = reinterpret_cast<uint64_t *>(
        reinterpret_cast<uintptr_t>(DataVar) +
        reinterpret_cast<uintptr_t>(DataVar->CounterPtr));
    for (size_t i = 0; i < NumCounters; i++)
      Counters[i] += Increments[i];
    break;
  }
}

static int RegisterRuntime() {
  __llvm_profile_initialize();
#ifdef _AIX
  extern COMPILER_RT_VISIBILITY void *__llvm_profile_keep[];
  (void)*(void *volatile *)__llvm_profile_keep;
#endif
  return 0;
}

/* int __llvm_profile_runtime  */
COMPILER_RT_VISIBILITY int INSTR_PROF_PROFILE_RUNTIME_VAR = RegisterRuntime();
}
