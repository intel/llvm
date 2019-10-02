//==---------------- cl_usm_ext.h - USM Extension for CL -------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************
 * cl_intel_unified_shared_memory extension *
 ********************************************/
#define cl_intel_unified_shared_memory 1

/* cl_device_info */
#define CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL 0x4190
#define CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL 0x4191
#define CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL 0x4192
#define CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL 0x4193
#define CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL 0x4194

// TODO: should this be cl_device_unified_shared_memory_capabilities_intel?
typedef cl_bitfield cl_unified_shared_memory_capabilities_intel;

/* cl_unified_shared_memory_capabilities_intel - bitfield */
#define CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL (1 << 0)
#define CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL (1 << 1)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL (1 << 2)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL (1 << 3)

typedef cl_bitfield cl_mem_properties_intel;

/* cl_mem_properties_intel */
#define CL_MEM_ALLOC_FLAGS_INTEL 0x4195

typedef cl_bitfield cl_mem_alloc_flags_intel;

/* cl_mem_alloc_flags_intel - bitfield */
#define CL_MEM_ALLOC_DEFAULT_INTEL 0
#define CL_MEM_ALLOC_WRITE_COMBINED_INTEL (1 << 0)

typedef cl_uint cl_mem_info_intel;

/* cl_mem_alloc_info_intel */
#define CL_MEM_ALLOC_TYPE_INTEL 0x419A
#define CL_MEM_ALLOC_BASE_PTR_INTEL 0x419B
#define CL_MEM_ALLOC_SIZE_INTEL 0x419C
#define CL_MEM_ALLOC_INFO_DEVICE_INTEL 0x419D
/* CL_MEM_ALLOC_FLAGS_INTEL - defined above */
#define CL_MEM_ALLOC_INFO_TBD1_INTEL 0x419E /* reserved for future */
#define CL_MEM_ALLOC_INFO_TBD2_INTEL 0x419F /* reserved for future */

typedef cl_uint cl_unified_shared_memory_type_intel;

/* cl_unified_shared_memory_type_intel */
#define CL_MEM_TYPE_UNKNOWN_INTEL 0x4196
#define CL_MEM_TYPE_HOST_INTEL 0x4197
#define CL_MEM_TYPE_DEVICE_INTEL 0x4198
#define CL_MEM_TYPE_SHARED_INTEL 0x4199

typedef cl_uint cl_mem_advice_intel;

/* cl_mem_advice_intel */
#define CL_MEM_ADVICE_TBD0_INTEL 0x4208 /* reserved for future */
#define CL_MEM_ADVICE_TBD1_INTEL 0x4209 /* reserved for future */
#define CL_MEM_ADVICE_TBD2_INTEL 0x420A /* reserved for future */
#define CL_MEM_ADVICE_TBD3_INTEL 0x420B /* reserved for future */
#define CL_MEM_ADVICE_TBD4_INTEL 0x420C /* reserved for future */
#define CL_MEM_ADVICE_TBD5_INTEL 0x420D /* reserved for future */
#define CL_MEM_ADVICE_TBD6_INTEL 0x420E /* reserved for future */
#define CL_MEM_ADVICE_TBD7_INTEL 0x420F /* reserved for future */

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL 0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL 0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL 0x4202
#define CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL 0x4203

/* cl_command_type */
#define CL_COMMAND_MEMSET_INTEL 0x4204
#define CL_COMMAND_MEMCPY_INTEL 0x4205
#define CL_COMMAND_MIGRATEMEM_INTEL 0x4206
#define CL_COMMAND_MEMADVISE_INTEL 0x4207

extern CL_API_ENTRY void *CL_API_CALL
clHostMemAllocINTEL(cl_context context, cl_mem_properties_intel *properties,
                    size_t size, cl_uint alignment, cl_int *errcode_ret);

typedef CL_API_ENTRY void *(CL_API_CALL *clHostMemAllocINTEL_fn)(
    cl_context context, cl_mem_properties_intel *properties, size_t size,
    cl_uint alignment, cl_int *errcode_ret);

extern CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret);

typedef CL_API_ENTRY void *(CL_API_CALL *clDeviceMemAllocINTEL_fn)(
    cl_context context, cl_device_id device,
    cl_mem_properties_intel *properties, // TBD: needed?
    size_t size, cl_uint alignment, cl_int *errcode_ret);

extern CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_properties_intel *properties, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret);

typedef CL_API_ENTRY void *(CL_API_CALL *clSharedMemAllocINTEL_fn)(
    cl_context context, cl_device_id device,
    cl_mem_properties_intel *properties, // TBD: needed?
    size_t size, cl_uint alignment, cl_int *errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clMemFreeINTEL(cl_context context,
               const void *ptr); // TBD: const?

typedef CL_API_ENTRY
cl_int(CL_API_CALL *clMemFreeINTEL_fn)(cl_context context,
                                       const void *ptr); // TBD: const?

extern CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

typedef CL_API_ENTRY cl_int(CL_API_CALL *clGetMemAllocInfoINTEL_fn)(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value);

typedef CL_API_ENTRY cl_int(CL_API_CALL *clSetKernelArgMemPointerINTEL_fn)(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value);

// TBD: Memset API vs. "fill" API.  Is this sufficient?
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(cl_command_queue command_queue, void *dst_ptr,
                     cl_int value, size_t size, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event);

// TBD: Memset API vs. "fill" API.  Is this sufficient?
typedef CL_API_ENTRY cl_int(CL_API_CALL *clEnqueueMemsetINTEL_fn)(
    cl_command_queue command_queue, void *dst_ptr, cl_int value, size_t size,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue command_queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

typedef CL_API_ENTRY cl_int(CL_API_CALL *clEnqueueMemcpyINTEL_fn)(
    cl_command_queue command_queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

typedef CL_API_ENTRY cl_int(CL_API_CALL *clEnqueueMigrateMemINTEL_fn)(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

typedef CL_API_ENTRY cl_int(CL_API_CALL *clEnqueueMemAdviseINTEL_fn)(
    cl_command_queue command_queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

#ifdef __cplusplus
}
#endif
