/*
 *
 * Copyright (C) 2021-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "ur_lib.h"
#include "ur_loader.h"

namespace ur_lib {

extern "C" BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason,
                                 LPVOID lpvReserved) {
    if (fdwReason == DLL_PROCESS_DETACH) {
        delete context;
        delete loader::context;
    } else if (fdwReason == DLL_PROCESS_ATTACH) {
        context = new context_t;
        loader::context = new loader::context_t;
    }
    return TRUE;
}

} // namespace ur_lib
