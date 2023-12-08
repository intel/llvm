/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "critnib/critnib.h"
#include "memory_tracker.h"

#include <windows.h>
#if defined(UMF_SHARED_LIBRARY)
critnib *TRACKER = NULL;
BOOL APIENTRY DllMain(HINSTANCE, DWORD fdwReason, LPVOID lpvReserved) {
    if (fdwReason == DLL_PROCESS_DETACH) {
        critnib_delete(TRACKER);
    } else if (fdwReason == DLL_PROCESS_ATTACH) {
        TRACKER = critnib_new();
    }
    return TRUE;
}
#else
struct tracker_t {
    tracker_t() { map = critnib_new(); }
    ~tracker_t() { critnib_delete(map); }
    critnib *map;
};
tracker_t TRACKER_INSTANCE;
critnib *TRACKER = TRACKER_INSTANCE.map;
#endif

umf_memory_tracker_handle_t umfMemoryTrackerGet(void) {
    return (umf_memory_tracker_handle_t)TRACKER;
}
