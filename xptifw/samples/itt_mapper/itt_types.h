/*
    Copyright 2014-2015 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#ifndef _ITT_TYPES_H_
#define _ITT_TYPES_H_

#include <cstring>
#include <iostream>

#ifndef ITT_OS_WIN
#define ITT_OS_WIN 1
#endif /* ITT_OS_WIN */

#ifndef ITT_OS_LINUX
#define ITT_OS_LINUX 2
#endif /* ITT_OS_LINUX */

#ifndef ITT_OS_MAC
#define ITT_OS_MAC 3
#endif /* ITT_OS_MAC */

#ifndef ITT_OS
#if defined WIN32 || defined _WIN32
#define ITT_OS ITT_OS_WIN
#elif defined(__APPLE__) && defined(__MACH__)
#define ITT_OS ITT_OS_MAC
#else
#define ITT_OS ITT_OS_LINUX
#endif
#endif /* ITT_OS */

#ifndef ITT_PLATFORM_WIN
#define ITT_PLATFORM_WIN 1
#endif /* ITT_PLATFORM_WIN */

#ifndef ITT_PLATFORM_POSIX
#define ITT_PLATFORM_POSIX 2
#endif /* ITT_PLATFORM_POSIX */

#ifndef ITT_PLATFORM
#if ITT_OS == ITT_OS_WIN
#define ITT_PLATFORM ITT_PLATFORM_WIN
#else
#define ITT_PLATFORM ITT_PLATFORM_POSIX
#endif /* _WIN32 */
#endif /* ITT_PLATFORM */

#if defined(_UNICODE) && !defined(UNICODE)
#define UNICODE
#endif

#include <stddef.h>
#if ITT_PLATFORM == ITT_PLATFORM_WIN
#include "windows.h"
#include <stdint.h>
#include <tchar.h>
#else /* ITT_PLATFORM==ITT_PLATFORM_WIN */
#include <stdint.h>
#if defined(UNICODE) || defined(_UNICODE)
#include <wchar.h>
#endif /* UNICODE || _UNICODE */
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

#define XML_MAGIC_v0 '<'
#define SIMPLE_BINARY_MAGIC_v0 'a'
#define OPTIMIZED_BINARY_MAGIC_v0 'A'

#if ITT_PLATFORM == ITT_PLATFORM_WIN
#include <intrin.h>

#if USE_OLD_PERF
namespace Perf {
typedef unsigned __int64 tick_t;
#if defined(_M_X64)
inline tick_t rdtsc() {
  unsigned int aux;
  return __rdtscp(&aux);
}
#elif _M_IX86
inline tick_t rdtsc() {
  __asm {
                lfence
                rdtsc
  }
}
#else
#error Unsupported ISA
#endif

inline uint64_t get_ts_frequency() {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart * 1000;
}
} // namespace Perf
#else
namespace Perf {
typedef unsigned __int64 tick_t;
inline tick_t rdtsc() {
  LARGE_INTEGER qpcnt;
  int rval = QueryPerformanceCounter(&qpcnt);
  return qpcnt.QuadPart;
}

inline uint64_t get_ts_frequency() {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart * 1000;
}

inline uint64_t get_cpu() { return GetCurrentProcessorNumber(); }
} // namespace Perf
#endif

#define FOPEN(o, f, m)                                                         \
  { fopen_s(&o, (f), (m)); }
#define WCSTOMBS(l, a, s, b, c)                                                \
  { wcstombs_s(&l, (a), (s), (b), (c)); }
#define STRTOUI64(s, e, b) _strtoui64((s), (e), (b))
#define STRDUP(s) _strdup((s))
#define WCSDUP(s) _wcsdup((s))

static inline bool MEMCPY_S(void *dest, size_t n, const void *src, size_t m) {
  if (!memcpy_s(dest, n, src, m)) {
    return true;
  } else {
    return false;
  }
}

typedef DWORD tls_key_t;
static inline void create_tls_key(tls_key_t &key) { key = TlsAlloc(); }
static inline void destroy_tls_key(tls_key_t key) { TlsFree(key); }
static inline void set_tls(tls_key_t key, void *value) {
  TlsSetValue(key, (LPVOID)value);
}
static inline void *get_tls(tls_key_t key) { return (void *)TlsGetValue(key); }

#elif __linux__ || __APPLE__

#include <cstring>
#include <stdint.h>
#include <unistd.h>
#include <wchar.h>

#if USE_OLD_PERF
namespace Perf {
typedef uint64_t tick_t;
#if __x86_64__ || __i386__ || __i386
inline tick_t rdtsc() {
  uint32_t lo, hi;
  __asm__ __volatile__("lfence\nrdtsc" : "=a"(lo), "=d"(hi));
  return (tick_t)lo | ((tick_t)hi) << 32;
}

inline uint64_t get_ts_frequency() {
  tick_t t0 = rdtsc();
  usleep(1000);
  tick_t t1 = rdtsc();
  return (t1 - t0) * 1000;
}
#else
#error Unsupported ISA
#endif

} // namespace Perf
#else
#include <sched.h>

namespace Perf {
typedef uint64_t tick_t;
#if __x86_64__ || __i386__ || __i386
inline tick_t rdtsc() {
  struct timespec ts;
  int status = clock_gettime(CLOCK_REALTIME, &ts);
  (void)status;
  return (static_cast<tick_t>(1000000000UL) * static_cast<tick_t>(ts.tv_sec) +
          static_cast<tick_t>(ts.tv_nsec));
}

inline uint64_t get_ts_frequency() { return static_cast<uint64_t>(1E9); }

inline uint64_t get_cpu() {
#ifdef __linux__
  return sched_getcpu();
#else
  return 0;
#endif
}
#else
#error Unsupported ISA
#endif
} // namespace Perf
#endif

#define FOPEN(o, f, m)                                                         \
  { o = fopen((f), (m)); }
#define WCSTOMBS(l, a, s, b, c)                                                \
  { l = wcstombs((a), (b), (c)); }
#define STRTOUI64(s, e, b) strtoull((s), (e), (b))
#define STRDUP(s) strdup((s))
#define WCSDUP(s) wcsdup((s))

static inline bool MEMCPY_S(void *d, size_t n, const void *s, size_t m) {
  if (m <= n) {
    std::memcpy(d, s, m);
    return true;
  } else {
    return false;
  }
}

typedef pthread_key_t tls_key_t;
static inline void create_tls_key(tls_key_t &key) {
  pthread_key_create(&key, NULL);
}
static inline void destroy_tls_key(tls_key_t key) { pthread_key_delete(key); }
static inline void set_tls(tls_key_t key, void *value) {
  pthread_setspecific(key, value);
}
static inline void *get_tls(tls_key_t key) { return pthread_getspecific(key); }

#else
#error Unsupported OS
#endif /* OS */

#pragma pack(push, 8)

typedef struct ___itt_domain {
  volatile int flags; /*!< Zero if disabled, non-zero if enabled. The meaning of
                         different non-zero values is reserved to the runtime */
  const char *nameA;  /*!< Copy of original name in ASCII. */
#if defined(UNICODE) || defined(_UNICODE)
  const wchar_t *nameW; /*!< Copy of original name in UNICODE. */
#else                   /* UNICODE || _UNICODE */
  void *nameW;
#endif                  /* UNICODE || _UNICODE */
  int extra1;           /*!< Reserved to the runtime */
  void *extra2;         /*!< Reserved to the runtime */
  struct ___itt_domain *next;
} __itt_domain;

#pragma pack(pop)

#pragma pack(push, 8)

typedef struct ___itt_id {
  unsigned long long d1, d2, d3;
} __itt_id;

#pragma pack(pop)

#pragma pack(push, 8)

typedef struct ___itt_string_handle {
  const char *strA; /*!< Copy of original string in ASCII. */
#if defined(UNICODE) || defined(_UNICODE)
  const wchar_t *strW; /*!< Copy of original string in UNICODE. */
#else                  /* UNICODE || _UNICODE */
  void *strW;
#endif                 /* UNICODE || _UNICODE */
  int extra1;          /*!< Reserved. Must be zero   */
  void *extra2;        /*!< Reserved. Must be zero   */
  struct ___itt_string_handle *next;
} __itt_string_handle;

#pragma pack(pop)

// #define TBB_STRING_RESOURCE(index_name, str) index_name,
// typedef enum {
//    #include "_tbb_strings.h"
//    NUM_STRINGS
// }
// TBB_string_index;
// #undef TBB_STRING_RESOURCE

typedef enum {
  __itt_relation_is_unknown = 0,
  __itt_relation_is_dependent_on, /**< "A is dependent on B" means that A cannot
                                     start until B completes */
  __itt_relation_is_sibling_of, /**< "A is sibling of B" means that A and B were
                                   created as a group */
  __itt_relation_is_parent_of, /**< "A is parent of B" means that A created B */
  __itt_relation_is_continuation_of, /**< "A is continuation of B" means that A
                                        assumes the dependencies of B */
  __itt_relation_is_child_of, /**< "A is child of B" means that A was created by
                                 B (inverse of is_parent_of) */
  __itt_relation_is_continued_by,  /**< "A is continued by B" means that B
                                      assumes the dependencies of A (inverse of
                                      is_continuation_of) */
  __itt_relation_is_predecessor_to /**< "A is predecessor to B" means that B
                                      cannot start until A completes (inverse of
                                      is_dependent_on) */
} __itt_relation;

typedef enum {
  __itt_metadata_unknown = 0,
  __itt_metadata_u64,   /**< Unsigned 64-bit integer */
  __itt_metadata_s64,   /**< Signed 64-bit integer */
  __itt_metadata_u32,   /**< Unsigned 32-bit integer */
  __itt_metadata_s32,   /**< Signed 32-bit integer */
  __itt_metadata_u16,   /**< Unsigned 16-bit integer */
  __itt_metadata_s16,   /**< Signed 16-bit integer */
  __itt_metadata_float, /**< Signed 32-bit floating-point */
  __itt_metadata_double /**< SIgned 64-bit floating-point */
} __itt_metadata_type;

// Add ostream for __itt_id
inline std::ostream &operator<<(std::ostream &o, __itt_id id) {
  return (o << id.d1 << ":" << id.d2 << ":" << id.d3);
}

typedef enum {
  _rt_null,
  _rt_unknown,
  _rt_id,      /** __itt_id type */
  _rt_u64,     /**< Unsigned 64-bit integer */
  _rt_s64,     /**< Signed 64-bit integer */
  _rt_u32,     /**< Unsigned 32-bit integer */
  _rt_s32,     /**< Signed 32-bit integer */
  _rt_u16,     /**< Unsigned 16-bit integer */
  _rt_s16,     /**< Signed 16-bit integer */
  _rt_u8,      /**< Unsigned 16-bit integer */
  _rt_s8,      /**< Signed 16-bit integer */
  _rt_float,   /**< Signed 32-bit floating-point */
  _rt_double,  /**< SIgned 64-bit floating-point */
  _rt_stringA, /**< character string */
  _rt_stringW  /**< wchar_t string */
} element_type;

const int NUM_FIELDS = 5;
typedef struct {
  const char *record_name;
  size_t base_size;
  element_type elem[NUM_FIELDS];
} record_desc_t;

typedef enum {
  REC_DEFINITION = 0,
  REC_API_VERSION,
  REC_DOMAIN_CREATE_A,
  REC_DOMAIN_CREATE_W,
  REC_ID_CREATE,
  REC_ID_DESTROY,
  REC_META_DATA,
  REC_META_DATA_STRING_A,
  REC_META_DATA_STRING_W,
  REC_RELATION,
  REC_STRING_HANDLE_CREATE_A,
  REC_STRING_HANDLE_CREATE_W,
  REC_TASK_BEGIN,
  REC_TASK_END,
  REC_TASK_GROUP,
  REC_REGION_BEGIN,
  REC_REGION_END,
  REC_END_OF_RECORDS,
  REC_BAD_RECORD
} __itt_record_t;

const record_desc_t record_descriptions[REC_END_OF_RECORDS + 1] = {
    {"definition",
     12 /* + stringAlen + numfields */,
     {/*op=*/_rt_u8, /*len=*/_rt_u8, /*stringA=*/_rt_stringA,
      /*numfields=*/_rt_u8, /*1 per field type[]=*/_rt_u8}},
    {"api_version",
     10 /* + stringAlen */,
     {/*len=*/_rt_u8, /*stringA=*/_rt_stringA, _rt_null, _rt_null, _rt_null}},
    {"domain_createA",
     18 /* + stringAlen */,
     {/*domain=*/_rt_u64, /*len=*/_rt_u8, /*stringA=*/_rt_stringA, _rt_null,
      _rt_null}},
    {"domain_createW",
     18 /* + stringWlen */,
     {/*domain=*/_rt_u64, /*len=*/_rt_u8, /*stringW=*/_rt_stringW, _rt_null,
      _rt_null}},
    {"id_create",
     41,
     {/*domain=*/_rt_u64, /*id=*/_rt_id, _rt_null, _rt_null, _rt_null}},
    {"id_destroy",
     41,
     {/*domain=*/_rt_u64, /*id=*/_rt_id, _rt_null, _rt_null, _rt_null}},
    {"metadata",
     58 /* + dataSize*count */,
     {/*domain=*/_rt_u64, /*id=*/_rt_id, /*key=*/_rt_u64, /*type=*/_rt_u8,
      /*count=*/_rt_u64}},
    {"metadata_stringA",
     57 /* + stringAlen */,
     {/*domain=*/_rt_u64, /*id=*/_rt_id, /*key=*/_rt_u64, /*len=*/_rt_u64,
      /*stringA=*/_rt_stringA}},
    {"metadata_stringW",
     57 /* + stringWlen */,
     {/*domain=*/_rt_u64, /*id=*/_rt_id, /*key=*/_rt_u64, /*len=*/_rt_u64,
      /*stringW=*/_rt_stringW}},
    {"relation",
     67,
     {/*domain=*/_rt_u64, /*head_id=*/_rt_id, /*relation=*/_rt_u8,
      /*tail_id=*/_rt_id, _rt_null}},
    {"string_handle_createA",
     18 /* + stringAlen */,
     {/*stringid=*/_rt_u64, /*len=*/_rt_u8, /*stringA=*/_rt_stringA, _rt_null,
      _rt_null}},
    {"string_handle_createW",
     18 /* + stringWlen */,
     {/*stringid=*/_rt_u64, /*len=*/_rt_u8, /*stringA=*/_rt_stringW, _rt_null,
      _rt_null}},
    {"task_begin",
     81,
     {/*domain=*/_rt_u64, /*taskid=*/_rt_id, /*parentid=*/_rt_id,
      /*stringhandle=*/_rt_u64, /*core_id*/ _rt_u64}},
    {"task_end",
     17,
     {/*domain=*/_rt_u64, _rt_null, _rt_null, _rt_null, _rt_null}},
    {"task_group",
     73,
     {/*domain=*/_rt_u64, /*groupid=*/_rt_id, /*parentid=*/_rt_id,
      /*stringhandle=*/_rt_u64, _rt_null}},
    {"region_begin",
     73,
     {/*domain=*/_rt_u64, /*regionid=*/_rt_id, /*parentid=*/_rt_id,
      /*stringhandle=*/_rt_u64, _rt_null}},
    {"region_end",
     41,
     {/*domain=*/_rt_u64, /*regionid=*/_rt_id, _rt_null, _rt_null, _rt_null}},
    {"end_of_records", 9, {_rt_null, _rt_null, _rt_null, _rt_null, _rt_null}}};
#endif
