/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
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
      UR_LOG(ERR,
             "Failed to unload the library with the handle at address 0x{}",
             handle);
    } else {
      UR_LOG(INFO, "unloaded adapter 0x{}", handle);
    }
  }
}

std::unique_ptr<HMODULE, LibLoader::lib_dtor>
LibLoader::loadAdapterLibrary(const char *name) {
  if (HMODULE handle = LoadLibraryExA(name, nullptr, 0)) {
    UR_LOG(INFO, "loaded adapter 0x{}: {}", handle, name);
    return std::unique_ptr<HMODULE, LibLoader::lib_dtor>{handle};
  } else {
    UR_LOG(DEBUG, "loading adapter failed with error {}: {}", GetLastError(),
           name);
  }
  return nullptr;
}

void *LibLoader::getFunctionPtr(HMODULE handle, const char *func_name) {
  return reinterpret_cast<void *>(GetProcAddress(handle, func_name));
}

} // namespace ur_loader
