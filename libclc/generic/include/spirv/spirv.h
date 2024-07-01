//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef cl_clang_storage_class_specifiers
#error Implementation requires cl_clang_storage_class_specifiers extension!
#endif

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
#define _CLC_GENERIC_AS_SUPPORTED 1
// Note that we hard-code the assumption that a non-distinct address space means
// that the target maps the generic address space to the private address space.
#if __CLC_DISTINCT_GENERIC_ADDRSPACE__
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 1
#else
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 0
#endif
#else
#define _CLC_GENERIC_AS_SUPPORTED 0
#define _CLC_DISTINCT_GENERIC_AS_SUPPORTED 0
#endif

/* Function Attributes */
#include <func.h>

/* Supported Data Types */
#include <types.h>
#include <spirv/spirv_types.h>

/* Supported builtins */
#include <spirv/spirv_builtins.h>

/* Reinterpreting Types Using as_type() and as_typen() */
#include <as_type.h>

/* Preprocessor Directives and Macros */
#include <macros.h>

/* 6.11.1 Work-Item Functions */
#include <spirv/workitem/get_global_id.h>
#include <spirv/workitem/get_global_offset.h>
#include <spirv/workitem/get_global_size.h>
#include <spirv/workitem/get_group_id.h>
#include <spirv/workitem/get_local_id.h>
#include <spirv/workitem/get_local_size.h>
#include <spirv/workitem/get_max_sub_group_size.h>
#include <spirv/workitem/get_num_groups.h>
#include <spirv/workitem/get_num_sub_groups.h>
#include <spirv/workitem/get_sub_group_id.h>
#include <spirv/workitem/get_sub_group_local_id.h>
#include <spirv/workitem/get_sub_group_size.h>
#include <spirv/workitem/get_work_dim.h>

/* 6.11.2.1 Floating-point macros */
#include <clc/float/definitions.h>

/* 6.11.3 Integer Definitions */
#include <clc/integer/definitions.h>

/* 6.11.11 Atomic Functions */
#include <spirv/atomic/atomic_add.h>
#include <spirv/atomic/atomic_and.h>
#include <spirv/atomic/atomic_cmpxchg.h>
#include <spirv/atomic/atomic_dec.h>
#include <spirv/atomic/atomic_inc.h>
#include <spirv/atomic/atomic_load.h>
#include <spirv/atomic/atomic_max.h>
#include <spirv/atomic/atomic_min.h>
#include <spirv/atomic/atomic_or.h>
#include <spirv/atomic/atomic_store.h>
#include <spirv/atomic/atomic_sub.h>
#include <spirv/atomic/atomic_xchg.h>
#include <spirv/atomic/atomic_xor.h>

/* 6.11.13 Image Read and Write Functions */
#include <spirv/image/image_defines.h>
#include <spirv/image/image.h>

#pragma OPENCL EXTENSION all : disable
