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
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef pthread_mutex_t os_mutex_t;

#define Malloc malloc
#define Free free

static void *Zalloc(size_t s) {
    void *m = Malloc(s);
    if (m) {
        memset(m, 0, s);
    }
    return m;
}

#define util_mutex_init(x) pthread_mutex_init(x, NULL)
#define util_mutex_destroy(x) pthread_mutex_destroy(x)
#define util_mutex_lock(x) pthread_mutex_lock(x)
#define util_mutex_unlock(x) pthread_mutex_unlock(x)
#define util_lssb_index64(x) ((unsigned char)__builtin_ctzll(x))
#define util_mssb_index64(x) ((unsigned char)(63 - __builtin_clzll(x)))
#define util_lssb_index32(x) ((unsigned char)__builtin_ctzl(x))
#define util_mssb_index32(x) ((unsigned char)(31 - __builtin_clzl(x)))
#if __SIZEOF_LONG__ == 8
#define util_lssb_index(x) util_lssb_index64(x)
#define util_mssb_index(x) util_mssb_index64(x)
#else
#define util_lssb_index(x) util_lssb_index32(x)
#define util_mssb_index(x) util_mssb_index32(x)
#endif

#define NOFUNCTION                                                             \
    do                                                                         \
        ;                                                                      \
    while (0)
#define VALGRIND_ANNOTATE_NEW_MEMORY(p, s) NOFUNCTION
#define VALGRIND_HG_DRD_DISABLE_CHECKING(p, s) NOFUNCTION

#ifdef NDEBUG
#define ASSERT(x) NOFUNCTION
#define ASSERTne(x, y) ASSERT(x != y)
#else
#include <stdio.h>
#define ASSERT(x)                                                              \
    do                                                                         \
        if (!(x)) {                                                            \
            fprintf(stderr,                                                    \
                    "Assertion failed: " #x " at " __FILE__ " line %d.\n",     \
                    __LINE__);                                                 \
            abort();                                                           \
        }                                                                      \
    while (0)
#define ASSERTne(x, y)                                                         \
    do {                                                                       \
        long X = (x);                                                          \
        long Y = (y);                                                          \
        if (X == Y) {                                                          \
            fprintf(stderr,                                                    \
                    "Assertion failed: " #x " != " #y                          \
                    ", both are %ld, at " __FILE__ " line %d.\n",              \
                    X, __LINE__);                                              \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif
