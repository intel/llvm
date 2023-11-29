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
 *  SYCL compatibility extension
 *
 *  kernel.hpp
 *
 *  Description:
 *    kernel functionality for the SYCL compatibility extension.
 **************************************************************************/

// The original source was under the license below:
//==---- kernel.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/info/info_desc.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>

namespace syclcompat {

typedef void (*kernel_functor)(sycl::queue &, const sycl::nd_range<3> &,
                               unsigned int, void **, void **);

struct kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      detail::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
}
static kernel_function_info get_kernel_function_info(const void *function) {
  kernel_function_info kernel_info;
  kernel_info.max_work_group_size =
      detail::dev_mgr::instance()
          .current_device()
          .get_info<sycl::info::device::max_work_group_size>();
  return kernel_info;
}

} // namespace syclcompat
