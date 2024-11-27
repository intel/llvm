/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_utils.cpp
 *
 */

#include "sanitizer_common/sanitizer_common.hpp"
#include "ur_sanitizer_layer.hpp"

#include <asm/param.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <gnu/lib-names.h>
#include <string>
#include <sys/mman.h>

extern "C" __attribute__((weak)) void __asan_init(void);

namespace ur_sanitizer_layer {

bool IsInASanContext() { return (void *)__asan_init != nullptr; }

uptr MmapNoReserve(uptr Addr, uptr Size) {
    Size = RoundUpTo(Size, EXEC_PAGESIZE);
    Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
    void *P = mmap((void *)Addr, Size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_NORESERVE | MAP_ANONYMOUS, -1, 0);
    return (uptr)P;
}

bool Munmap(uptr Addr, uptr Size) { return munmap((void *)Addr, Size) == 0; }

bool DontCoredumpRange(uptr Addr, uptr Size) {
    Size = RoundUpTo(Size, EXEC_PAGESIZE);
    Addr = RoundDownTo(Addr, EXEC_PAGESIZE);
    return madvise((void *)Addr, Size, MADV_DONTDUMP) == 0;
}

void *GetMemFunctionPointer(const char *FuncName) {
    void *handle = dlopen(LIBC_SO, RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) {
        getContext()->logger.error("Failed to dlopen {}", LIBC_SO);
        return nullptr;
    }
    auto ptr = dlsym(handle, FuncName);
    if (!ptr) {
        getContext()->logger.error("Failed to get '{}' from {}", FuncName,
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
