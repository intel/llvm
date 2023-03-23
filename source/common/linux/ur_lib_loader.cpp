/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <dlfcn.h>

#include "logger/ur_logger.hpp"
#include "ur_lib_loader.hpp"

#if defined(SANITIZER_ANY)
#define LOAD_DRIVER_LIBRARY(NAME) dlopen(NAME, RTLD_LAZY | RTLD_LOCAL)
#else
#define LOAD_DRIVER_LIBRARY(NAME)                                              \
    dlopen(NAME, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)
#endif

namespace loader {

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
    return std::unique_ptr<HMODULE, LibLoader::lib_dtor>(
        LOAD_DRIVER_LIBRARY(name));
}

void *LibLoader::getFunctionPtr(HMODULE handle, const char *func_name) {
    return dlsym(handle, func_name);
}

} // namespace loader
