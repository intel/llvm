/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_utils.cpp
 *
 */

#include "sanitizer_common/sanitizer_common.hpp"
#include "ur_sanitizer_layer.hpp"

#include <asm/param.h>
#include <cerrno>
#include <cstring>
#include <cxxabi.h>
#include <dlfcn.h>
#include <elf.h>
#include <gnu/lib-names.h>
#include <string>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <unistd.h>

extern "C" __attribute__((weak)) void __asan_init(void);

namespace ur_sanitizer_layer {

bool IsInASanContext() { return (void *)__asan_init != nullptr; }

bool MmapFixedNoReserve(uptr Addr, uptr Size) {
  Size = RoundUpTo(Size, EXEC_PAGESIZE);
  Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
  return mmap((void *)Addr, Size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_FIXED_NOREPLACE | MAP_NORESERVE | MAP_ANONYMOUS,
              -1, 0) != MAP_FAILED;
}

uptr MmapNoReserve(uptr Addr, uptr Size) {
  Size = RoundUpTo(Size, EXEC_PAGESIZE);
  Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
  void *P = mmap((void *)Addr, Size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_NORESERVE | MAP_ANONYMOUS, -1, 0);
  if (P == MAP_FAILED) {
    return 0;
  }
  return (uptr)P;
}

bool Munmap(uptr Addr, uptr Size) { return munmap((void *)Addr, Size) == 0; }

bool ProtectMemoryRange(uptr Addr, uptr Size) {
  Size = RoundUpTo(Size, EXEC_PAGESIZE);
  Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
  return mmap((void *)Addr, Size, PROT_NONE,
              MAP_PRIVATE | MAP_FIXED_NOREPLACE | MAP_NORESERVE | MAP_ANONYMOUS,
              -1, 0) != MAP_FAILED;
}

bool DontCoredumpRange(uptr Addr, uptr Size) {
  Size = RoundUpTo(Size, EXEC_PAGESIZE);
  Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
  return madvise((void *)Addr, Size, MADV_DONTDUMP) == 0;
}

extern "C" {
__attribute__((weak)) extern void *__libc_stack_end;
}

static void GetArgsAndEnv(char ***Argv, char ***Envp) {
  if (&__libc_stack_end) {
    uptr *StackEnd = (uptr *)__libc_stack_end;
    int argc = *StackEnd;
    *Argv = (char **)(StackEnd + 1);
    *Envp = (char **)(StackEnd + argc + 2);
  } else {
    die("Can't get arguments and environment variables, possibly due to "
        "incompatible libc.");
  }
}

static void ReExec() {
  const char *PathName = reinterpret_cast<const char *>(getauxval(AT_EXECFN));
  char **Argv, **Envp;
  GetArgsAndEnv(&Argv, &Envp);
  execve(PathName, Argv, Envp);
  std::string Err = "ReExec failed: " + std::string(strerror(errno)) + ".\n";
  die(Err.c_str());
}

void TryReExecWithoutASLR() {
  // If failed to reserve shadow memory, check if ASLR is on. If ASLR is on,
  // re-exec with ASLR off.
  int OldPersonality = personality(0xffffffff);
  if ((OldPersonality != -1) && ((OldPersonality & ADDR_NO_RANDOMIZE) == 0)) {
    UR_LOG_L(getContext()->logger, DEBUG,
             "memory layout is incompatible, possibly due to high-entropy "
             "ASLR. Re-execing with fixed virtual address space.");
    if (personality(OldPersonality | ADDR_NO_RANDOMIZE) == -1) {
      die("Unable to disable ASLR, Device Sanitizer can't work properly.");
    }
    ReExec();
  }
}

void *GetMemFunctionPointer(const char *FuncName) {
  void *handle = dlopen(LIBC_SO, RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to dlopen {}", LIBC_SO);
    return nullptr;
  }
  auto ptr = dlsym(handle, FuncName);
  if (!ptr) {
    UR_LOG_L(getContext()->logger, ERR, "Failed to get '{}' from {}", FuncName,
             LIBC_SO);
  }
  return ptr;
}

std::string DemangleName(const std::string &name) {
  std::string result = name;
  char *demangled =
      abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr);
  if (demangled) {
    result = demangled;
    free(demangled);
  }
  return result;
}

} // namespace ur_sanitizer_layer
