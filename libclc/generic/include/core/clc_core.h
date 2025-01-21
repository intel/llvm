//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains some "core" function common to SPIR-V and OpenCL.
// Some SPIR-V builtins has a "s" or "u" prefix
// (depending on the sign of the operands).
// This is not useful from the libclc point of view and
// add extra complexity to the SPIR-V side.
//
// Core function are prefixed by __clc_

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

#include <as_type.h>
#include <clc/clcfunc.h>
#include <clc/clctypes.h>
#include <macros.h>

#include <clc/float/definitions.h>
#include <clc/integer/definitions.h>

#include <core/convert.h>

#include <core/integer/clc_abs.h>
#include <core/integer/clc_abs_diff.h>
#include <core/integer/clc_add_sat.h>
#include <core/integer/clc_hadd.h>
#include <core/integer/clc_mad24.h>
#include <core/integer/clc_mad_hi.h>
#include <core/integer/clc_mad_sat.h>
#include <core/integer/clc_mul24.h>
#include <core/integer/clc_mul_hi.h>
#include <core/integer/clc_rhadd.h>
#include <core/integer/clc_sub_sat.h>
#include <core/integer/clc_upsample.h>

#include <core/shared/clc_clamp.h>
#include <core/shared/clc_max.h>
#include <core/shared/clc_min.h>

#pragma OPENCL EXTENSION all : disable
