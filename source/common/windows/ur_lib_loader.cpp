/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "ur_lib_loader.hpp"
#include "logger/ur_logger.hpp"

namespace ur_loader {

void LibLoader::freeAdapterLibrary(HMODULE handle) {
    if (handle) {
        BOOL res = FreeLibrary(handle);
        if (!res) {
            logger::error(
                "Failed to unload the library with the handle at address {}",
                handle);
        } else {
            logger::info("unloaded adapter 0x{}", handle);
        }
    }
}

std::unique_ptr<HMODULE, LibLoader::lib_dtor>
LibLoader::loadAdapterLibrary(const char *name) {
    auto handle = std::unique_ptr<HMODULE, LibLoader::lib_dtor>(
        LoadLibraryExA(name, nullptr, 0));
    logger::info("loaded adapter 0x{} ({})", handle, name);
    return handle;
}

void *LibLoader::getFunctionPtr(HMODULE handle, const char *func_name) {
    return reinterpret_cast<void *>(GetProcAddress(handle, func_name));
}

} // namespace ur_loader
