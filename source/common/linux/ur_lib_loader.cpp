/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include <dlfcn.h>

#include "logger/ur_logger.hpp"
#include "ur_lib_loader.hpp"

#define DEEP_BIND_ENV "UR_ADAPTERS_DEEP_BIND"

namespace ur_loader {

void LibLoader::freeAdapterLibrary(HMODULE handle) {
    if (handle) {
        int res = dlclose(handle);
        if (res) {
            logger::error(
                "Failed to unload the library with the handle at address {}",
                handle);
        }
    }
}

std::unique_ptr<HMODULE, LibLoader::lib_dtor>
LibLoader::loadAdapterLibrary(const char *name) {
    int mode = RTLD_LAZY | RTLD_LOCAL;
#if !defined(__APPLE__)
    bool deepbind = getenv_tobool(DEEP_BIND_ENV);
    if (deepbind) {
#if defined(SANITIZER_ANY)
        logger::warning(
            "Enabling RTLD_DEEPBIND while running under a sanitizer is likely "
            "to cause issues. Consider disabling {} environment variable.",
            DEEP_BIND_ENV);
#endif
        mode |= RTLD_DEEPBIND;
    }
#endif

    return std::unique_ptr<HMODULE, LibLoader::lib_dtor>(dlopen(name, mode));
}

void *LibLoader::getFunctionPtr(HMODULE handle, const char *func_name) {
    return dlsym(handle, func_name);
}

} // namespace ur_loader
