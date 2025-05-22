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

/* Function Attributes */
#include <clc/clcfunc.h>

/* Supported Data Types */
#include <clc/clctypes.h>
#include <libspirv/spirv_types.h>

/* Supported builtins */
#include <libspirv/spirv_builtins.h>

/* Reinterpreting Types Using __clc_as_type() and __clc_as_typen() */
#include <clc/clc_as_type.h>

/* 6.11.1 Work-Item Functions */
#include <libspirv/workitem/get_global_id.h>
#include <libspirv/workitem/get_global_offset.h>
#include <libspirv/workitem/get_global_size.h>
#include <libspirv/workitem/get_group_id.h>
#include <libspirv/workitem/get_local_id.h>
#include <libspirv/workitem/get_local_linear_id.h>
#include <libspirv/workitem/get_local_size.h>
#include <libspirv/workitem/get_max_sub_group_size.h>
#include <libspirv/workitem/get_num_groups.h>
#include <libspirv/workitem/get_num_sub_groups.h>
#include <libspirv/workitem/get_sub_group_id.h>
#include <libspirv/workitem/get_sub_group_local_id.h>
#include <libspirv/workitem/get_sub_group_size.h>
#include <libspirv/workitem/get_work_dim.h>

/* 6.11.2.1 Floating-point macros */
#include <clc/float/definitions.h>

/* 6.11.3 Integer Definitions */
#include <clc/integer/definitions.h>

/* 6.11.11 Atomic Functions */
#include <libspirv/atomic/atomic_add.h>
#include <libspirv/atomic/atomic_and.h>
#include <libspirv/atomic/atomic_cmpxchg.h>
#include <libspirv/atomic/atomic_dec.h>
#include <libspirv/atomic/atomic_inc.h>
#include <libspirv/atomic/atomic_load.h>
#include <libspirv/atomic/atomic_max.h>
#include <libspirv/atomic/atomic_min.h>
#include <libspirv/atomic/atomic_or.h>
#include <libspirv/atomic/atomic_store.h>
#include <libspirv/atomic/atomic_sub.h>
#include <libspirv/atomic/atomic_xchg.h>
#include <libspirv/atomic/atomic_xor.h>

/* 6.11.13 Image Read and Write Functions */
#include <libspirv/image/image.h>
#include <libspirv/image/image_defines.h>

/* Pointer Conversion */
#include <libspirv/conversion/GenericCastToPtrExplicit.h>

#pragma OPENCL EXTENSION all : disable
