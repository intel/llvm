/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <mutex>

#include "utils.h"

struct os_mutex_t *util_mutex_create(void) {
    return reinterpret_cast<struct os_mutex_t *>(new std::mutex);
}

void util_mutex_destroy(struct os_mutex_t *mutex) {
    delete reinterpret_cast<std::mutex *>(mutex);
}

int util_mutex_lock(struct os_mutex_t *mutex) try {
    reinterpret_cast<std::mutex *>(mutex)->lock();
    return 0;
} catch (std::system_error &err) {
    return err.code().value();
}

int util_mutex_unlock(struct os_mutex_t *mutex) {
    reinterpret_cast<std::mutex *>(mutex)->unlock();
    return 0;
}
