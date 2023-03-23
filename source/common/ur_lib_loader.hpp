/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef UR_LIB_LOADER_HPP
#define UR_LIB_LOADER_HPP 1

#include <memory>

#include "ur_util.hpp"

namespace loader {

class LibLoader {
  public:
    struct lib_dtor {
        typedef HMODULE pointer;
        void operator()(HMODULE handle) { freeAdapterLibrary(handle); }
    };

    static std::unique_ptr<HMODULE, lib_dtor>
    loadAdapterLibrary(const char *name);

    static void freeAdapterLibrary(HMODULE handle);

    static void *getFunctionPtr(HMODULE handle, const char *func_name);
};

} // namespace loader

#endif // UR_LIB_LOADER_HPP
