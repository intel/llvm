/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct os_mutex_t;

struct os_mutex_t *util_mutex_create(void);
void util_mutex_destroy(struct os_mutex_t *mutex);
int util_mutex_lock(struct os_mutex_t *mutex);
int util_mutex_unlock(struct os_mutex_t *mutex);

#if defined(_WIN32)
static __inline unsigned char util_lssb_index(long long value) {
    unsigned long ret;
    _BitScanForward64(&ret, value);
    return (unsigned char)ret;
}
static __inline unsigned char util_mssb_index(long long value) {
    unsigned long ret;
    _BitScanReverse64(&ret, value);
    return (unsigned char)ret;
}

// There is no good way to do atomic_load on windows...
#define util_atomic_load_acquire(object, dest)                                 \
    do {                                                                       \
        *dest = InterlockedOr64Acquire((LONG64 volatile *)object, 0);          \
    } while (0)

#define util_atomic_store_release(object, desired)                             \
    InterlockedExchange64((LONG64 volatile *)object, (LONG64)desired)
#define util_atomic_increment(object)                                          \
    InterlockedIncrement64((LONG64 volatile *)object)
#else
#define util_lssb_index(x) ((unsigned char)__builtin_ctzll(x))
#define util_mssb_index(x) ((unsigned char)(63 - __builtin_clzll(x)))
#define util_atomic_load_acquire(object, dest)                                 \
    __atomic_load(object, dest, memory_order_acquire)
#define util_atomic_store_release(object, desired)                             \
    __atomic_store_n(object, desired, memory_order_release)
#define util_atomic_increment(object)                                          \
    __atomic_add_fetch(object, 1, __ATOMIC_ACQ_REL)
#endif

#define Malloc malloc
#define Free free

static inline void *Zalloc(size_t s) {
    void *m = Malloc(s);
    if (m) {
        memset(m, 0, s);
    }
    return m;
}

#define NOFUNCTION                                                             \
    do {                                                                       \
    } while (0)
#define VALGRIND_ANNOTATE_NEW_MEMORY(p, s) NOFUNCTION
#define VALGRIND_HG_DRD_DISABLE_CHECKING(p, s) NOFUNCTION

#ifdef NDEBUG
#define ASSERT(x) NOFUNCTION
#define ASSERTne(x, y) ASSERT(x != y)
#else
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

#ifdef __cplusplus
}
#endif
