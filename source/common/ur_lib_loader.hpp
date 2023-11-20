/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#ifndef UR_LIB_LOADER_HPP
#define UR_LIB_LOADER_HPP 1

#include <memory>

#if _WIN32
#include <windows.h>
#else
#define HMODULE void *
#endif

namespace ur_loader {

class LibLoader {
  public:
    struct lib_dtor {
        typedef HMODULE pointer;
        void operator()(HMODULE handle) { freeAdapterLibrary(handle); }
    };

    using Lib = std::unique_ptr<HMODULE, lib_dtor>;

    static Lib loadAdapterLibrary(const char *name);

    static void freeAdapterLibrary(HMODULE handle);

    static void *getFunctionPtr(HMODULE handle, const char *func_name);
};

} // namespace ur_loader

#endif // UR_LIB_LOADER_HPP
