/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat
 *
 *  defs.hpp
 *
 *  Description:
 *    helper aliases and definitions for SYCLcompat
 *
 **************************************************************************/

// The original source was under the license below:
//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

template <class... Args> class sycl_compat_kernel_name;
template <int Arg> class sycl_compat_kernel_scalar;

#define __sycl_compat_align__(n) alignas(n)
#define __sycl_compat_inline__ __inline__ __attribute__((always_inline))

#define __sycl_compat_noinline__ __attribute__((noinline))

#define SYCL_COMPAT_COMPATIBILITY_TEMP (600)
