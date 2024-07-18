//==---------------- types.hpp --- SYCL types ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_lists.hpp>  // for vector_basic_list
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/is_device_copyable.hpp>
#include <sycl/detail/type_list.hpp>           // for is_contained
#include <sycl/detail/type_traits.hpp>         // for is_floating_point
#include <sycl/exception.hpp>                  // for make_error_code, errc
#include <sycl/half_type.hpp>                  // for StorageT, half, Vec16...
#include <sycl/marray.hpp>                     // for __SYCL_BINOP, __SYCL_...
#include <sycl/multi_ptr.hpp>                  // for multi_ptr
#include <sycl/vector_preview.hpp>             // for sycl::vec and swizzles

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16
