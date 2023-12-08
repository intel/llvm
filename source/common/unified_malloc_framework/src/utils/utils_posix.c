/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <pthread.h>
#include <stdlib.h>

#include "utils.h"

struct os_mutex_t *util_mutex_create(void) {
    pthread_mutex_t *mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
    int ret = pthread_mutex_init(mutex, NULL);
    return ret == 0 ? ((struct os_mutex_t *)mutex) : NULL;
}

void util_mutex_destroy(struct os_mutex_t *m) {
    pthread_mutex_t *mutex = (pthread_mutex_t *)m;
    int ret = pthread_mutex_destroy(mutex);
    (void)ret; // TODO: add logging
    free(m);
}

int util_mutex_lock(struct os_mutex_t *m) {
    return pthread_mutex_lock((pthread_mutex_t *)m);
}

int util_mutex_unlock(struct os_mutex_t *m) {
    return pthread_mutex_unlock((pthread_mutex_t *)m);
}
