//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "itt_minimal.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

const char *__itt_api_version(void) { return "0.1"; }

__itt_domain *__itt_domain_createA(const char *stringA) { return nullptr; }

#if (ITT_PLATFORM == ITT_PLATFORM_WIN) &&                                      \
    (defined(UNICODE) || defined(_UNICODE))
__itt_domain *__itt_domain_create(const wchar_t *stringW) { return nullptr; }

__itt_domain *__itt_domain_createW(const wchar_t *stringW) { return nullptr; }
#else
__itt_domain *__itt_domain_create(const char *stringA) { return nullptr; }
#endif

void __itt_id_create(const __itt_domain *domain, __itt_id id) {}

void __itt_id_destroy(const __itt_domain *domain, __itt_id id) {}

void __itt_metadata_str_addA(const __itt_domain *domain, __itt_id id,
                             __itt_string_handle *key, const char *stringA,
                             size_t len) {}

#if (ITT_PLATFORM == ITT_PLATFORM_WIN) &&                                      \
    (defined(UNICODE) || defined(_UNICODE))
void __itt_metadata_str_add(const __itt_domain *domain, __itt_id id,
                            __itt_string_handle *key, const wchar_t *stringW,
                            size_t len) {}

void __itt_metadata_str_addW(const __itt_domain *domain, __itt_id id,
                             __itt_string_handle *key, const wchar_t *stringW,
                             size_t len) {}

#else
void __itt_metadata_str_add(const __itt_domain *domain, __itt_id id,
                            __itt_string_handle *key, const char *stringA,
                            size_t len) {}
#endif

void __itt_metadata_add(const __itt_domain *domain, __itt_id id,
                        __itt_string_handle *key, __itt_metadata_type type,
                        size_t count, void *data) {}

void __itt_relation_add(const __itt_domain *domain, __itt_id head,
                        __itt_relation relation, __itt_id tail) {}

__itt_string_handle *__itt_string_handle_createA(const char *stringA) {
  return nullptr;
}

#if (ITT_PLATFORM == ITT_PLATFORM_WIN) &&                                      \
    (defined(UNICODE) || defined(_UNICODE))
__itt_string_handle *__itt_string_handle_create(const wchar_t *stringW) {
  return nullptr;
}

__itt_string_handle *__itt_string_handle_createW(const wchar_t *stringW) {
  return nullptr;
}
#else
__itt_string_handle *__itt_string_handle_create(const char *stringA) {
  return nullptr;
}
#endif

void __itt_task_begin(const __itt_domain *domain, __itt_id taskid,
                      __itt_id parentid, __itt_string_handle *name) {}

void __itt_task_end(const __itt_domain *domain) {}

void __itt_region_begin(const __itt_domain *domain, __itt_id regionid,
                        __itt_id parentid, __itt_string_handle *name) {}

void __itt_region_end(const __itt_domain *domain, __itt_id regionid) {}

void __itt_task_group(const __itt_domain *domain, __itt_id groupid,
                      __itt_id parentid, __itt_string_handle *name) {}

void __itt_api_fini(void *) {}

#ifdef __cplusplus
}
#endif
