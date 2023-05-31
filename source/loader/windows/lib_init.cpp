/*
 *
 * Copyright (C) 2021-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "ur_lib.hpp"
#include "ur_loader.hpp"

namespace ur_lib {

extern "C" BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason,
                                 LPVOID lpvReserved) {
    if (fdwReason == DLL_PROCESS_DETACH) {
        delete context;
        delete ur_loader::context;
    } else if (fdwReason == DLL_PROCESS_ATTACH) {
        context = new context_t;
        ur_loader::context = new ur_loader::context_t;
    }
    return TRUE;
}

} // namespace ur_lib
