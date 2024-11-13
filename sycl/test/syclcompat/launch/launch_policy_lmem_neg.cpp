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
 *  SYCLcompat API
 *
 *  launch_policy_lmem_neg.cpp
 *
 *  Description:
 *     Negative testing for launch_policy - local memory specific
 *     These tests are in their own TU because they instantiate some of the same
 *     templates as tests in launch_policy_neg.cpp
 **************************************************************************/

// RUN: not %clangxx -fsycl -fsyntax-only %s 2>&1 | FileCheck -vv %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>

// Dummy kernels for testing
inline void int_kernel(int a){};
inline void dynamic_local_mem_empty_kernel(char *a){};

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;

void test_lmem_launch() {
  sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};

  // Missing local mem
  {
    compat_exp::launch_policy policy(
        launch_range,
        compat_exp::kernel_properties{sycl_exp::sub_group_size<32>});
    compat_exp::launch<dynamic_local_mem_empty_kernel>(policy);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }

  // Unneeded local mem
  {
    compat_exp::launch_policy lmem_policy(launch_range,
                                          compat_exp::local_mem_size{1024});
    int int_arg{1};
    compat_exp::launch<int_kernel>(lmem_policy, int_arg);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }
}
