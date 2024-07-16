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

#include <iostream>

template <class... Args> class syclcompat_kernel_name;
template <int Arg> class syclcompat_kernel_scalar;

#if defined(_MSC_VER)
#define __syclcompat_align__(n) __declspec(align(n))
#define __syclcompat_inline__ __forceinline
#define __syclcompat_noinline__ __declspec(noinline)
#else
#define __syclcompat_align__(n) __attribute__((aligned(n)))
#define __syclcompat_inline__ __inline__ __attribute__((always_inline))
#define __syclcompat_noinline__ __attribute__((noinline))
#endif

#define SYCLCOMPAT_COMPATIBILITY_TEMP (600)

#ifdef _WIN32
#define SYCLCOMPAT_EXPORT __declspec(dllexport)
#else
#define SYCLCOMPAT_EXPORT
#endif

#define SYCLCOMPAT_MAJOR_VERSION 0
#define SYCLCOMPAT_MINOR_VERSION 1
#define SYCLCOMPAT_PATCH_VERSION 0

#define SYCLCOMPAT_MAKE_VERSION(_major, _minor, _patch)                        \
  ((1E6 * _major) + (1E3 * _minor) + _patch)

#define SYCLCOMPAT_VERSION                                                     \
  SYCLCOMPAT_MAKE_VERSION(SYCLCOMPAT_MAJOR_VERSION, SYCLCOMPAT_MINOR_VERSION,  \
                          SYCLCOMPAT_PATCH_VERSION)

namespace syclcompat {
enum error_code { SUCCESS = 0, BACKEND_ERROR = 1, DEFAULT_ERROR = 999 };
}

#define SYCLCOMPAT_CHECK_ERROR(expr)                                           \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return syclcompat::error_code::SUCCESS;                                  \
    } catch (sycl::exception const &e) {                                       \
      std::cerr << e.what() << std::endl;                                      \
      return syclcompat::error_code::BACKEND_ERROR;                            \
    } catch (std::runtime_error const &e) {                                    \
      std::cerr << e.what() << std::endl;                                      \
      return syclcompat::error_code::DEFAULT_ERROR;                            \
    }                                                                          \
  }()
