//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2025 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#ifdef __CLC_GEN_S

#ifdef __CLC_ATOMIC_MIN
#define __CLC_FUNCTION __spirv_AtomicSMin
#else
#define __CLC_FUNCTION __spirv_AtomicSMax
#endif

#else

#ifdef __CLC_ATOMIC_MIN
#define __CLC_FUNCTION __spirv_AtomicUMin
#else
#define __CLC_FUNCTION __spirv_AtomicUMax
#endif

#endif

#include <atomic_def.inc>

#undef __CLC_FUNCTION
