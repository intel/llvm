/*
    Copyright 2014=2015 Intel Corporation.  All Rights Reserved.

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

#ifndef _ITT_MINIMAL_H_
#define _ITT_MINIMAL_H_

#include "itt_types.h"

#if _MSC_VER >= 1400
#define ITT_TBB_EXPORT __declspec(dllexport)
#else
#define ITT_TBB_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if ITT_PLATFORM == ITT_PLATFORM_WIN
ITT_TBB_EXPORT __itt_domain *__itt_domain_createA(const char *name);
ITT_TBB_EXPORT __itt_domain *__itt_domain_createW(const wchar_t *name);
ITT_TBB_EXPORT __itt_string_handle *
__itt_string_handle_createA(const char *name);
ITT_TBB_EXPORT __itt_string_handle *
__itt_string_handle_createW(const wchar_t *name);
ITT_TBB_EXPORT void __itt_metadata_str_addA(const __itt_domain *domain,
                                            __itt_id id,
                                            __itt_string_handle *key,
                                            const char *data, size_t length);
ITT_TBB_EXPORT void __itt_metadata_str_addW(const __itt_domain *domain,
                                            __itt_id id,
                                            __itt_string_handle *key,
                                            const wchar_t *data, size_t length);
#endif /* ITT_PLATFORM==ITT_PLATFORM_WIN */

#if (ITT_PLATFORM == ITT_PLATFORM_WIN) &&                                      \
    (defined(UNICODE) || defined(_UNICODE))
ITT_TBB_EXPORT __itt_domain *__itt_domain_create(const wchar_t *name);
ITT_TBB_EXPORT __itt_string_handle *
__itt_string_handle_create(const wchar_t *name);
ITT_TBB_EXPORT void __itt_metadata_str_add(const __itt_domain *domain,
                                           __itt_id id,
                                           __itt_string_handle *key,
                                           const wchar_t *data, size_t length);
#else
ITT_TBB_EXPORT __itt_domain *__itt_domain_create(const char *name);
ITT_TBB_EXPORT __itt_string_handle *
__itt_string_handle_create(const char *name);
ITT_TBB_EXPORT void __itt_metadata_str_add(const __itt_domain *domain,
                                           __itt_id id,
                                           __itt_string_handle *key,
                                           const char *data, size_t length);
#endif

ITT_TBB_EXPORT void __itt_id_create(const __itt_domain *domain, __itt_id id);
ITT_TBB_EXPORT void __itt_id_destroy(const __itt_domain *domain, __itt_id id);
ITT_TBB_EXPORT const char *__itt_api_version(void);
ITT_TBB_EXPORT void __itt_task_group(const __itt_domain *domain, __itt_id id,
                                     __itt_id parentid,
                                     __itt_string_handle *name);
ITT_TBB_EXPORT void __itt_task_begin(const __itt_domain *domain,
                                     __itt_id taskid, __itt_id parentid,
                                     __itt_string_handle *name);
ITT_TBB_EXPORT void __itt_task_end(const __itt_domain *domain);
ITT_TBB_EXPORT void __itt_relation_add(const __itt_domain *domain,
                                       __itt_id head, __itt_relation relation,
                                       __itt_id tail);
ITT_TBB_EXPORT void __itt_region_begin(const __itt_domain *domain,
                                       __itt_id taskid, __itt_id parentid,
                                       __itt_string_handle *name);
ITT_TBB_EXPORT void __itt_region_end(const __itt_domain *domain,
                                     __itt_id taskid);
ITT_TBB_EXPORT void __itt_api_fini(void *);
ITT_TBB_EXPORT void __itt_metadata_add(const __itt_domain *domain, __itt_id id,
                                       __itt_string_handle *key,
                                       __itt_metadata_type type, size_t count,
                                       void *data);
#ifdef __cplusplus
}
#endif

#endif
