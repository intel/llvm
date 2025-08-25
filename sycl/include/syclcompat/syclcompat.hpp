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
 *  syclcompat.hpp
 *
 *  Description:
 *    Main include internal header for SYCLcompat
 **************************************************************************/

#pragma once

// MSVC ignores [[deprecated]] attribute on namespace unless compiled with
// /W3 or above.
#ifdef _MSC_VER
#define __SYCLCOMPAT_STRINGIFY(x) #x
#define __SYCLCOMPAT_TOSTRING(x) __SYCLCOMPAT_STRINGIFY(x)

#define __SYCLCOMPAT_WARNING(msg)                                              \
  __pragma(message(__FILE__                                                    \
                   "(" __SYCLCOMPAT_TOSTRING(__LINE__) "): warning: " msg))

__SYCLCOMPAT_WARNING("syclcompat is deprecated and the deprecation warnings "
                     "are ignored unless compiled with /W3 or above.")

#undef __SYCLCOMPAT_WARNING
#undef __SYCLCOMPAT_TOSTRING
#undef __SYCLCOMPAT_STRINGIFY
#endif

#include <syclcompat/atomic.hpp>
#include <syclcompat/defs.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/group_utils.hpp>
#include <syclcompat/id_query.hpp>
#include <syclcompat/kernel.hpp>
#include <syclcompat/launch.hpp>
#include <syclcompat/math.hpp>
#include <syclcompat/memory.hpp>
#include <syclcompat/util.hpp>
