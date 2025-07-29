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
 *  kernel_properties.cpp
 *
 *  Description:
 *     launch<F> with kernel_properties tests
 **************************************************************************/

// We need hardware which can support at least 2 sub-group sizes, since that
// hardware (presumably) supports the `intel_reqd_sub_group_size` attribute.
// REQUIRES: sg-32 && sg-16
// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-is-device %if cl_options %{/clang:-S /clang:-emit-llvm%} %else %{-S -emit-llvm%} %s -o - | FileCheck %s
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <syclcompat/launch.hpp>

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;

// Dummy kernel function for testing
inline void empty_kernel_1(){};
inline void empty_kernel_2(){};

// Set `sub_group_size` property for kernel & check it becomes attribute
// `reqd_sub_group_size`
int test_kernel_properties() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  compat_exp::kernel_properties my_k_props{sycl_exp::sub_group_size<32>};
  compat_exp::launch_policy my_config(sycl::nd_range<1>{{32}, {32}},
                                      my_k_props);
  compat_exp::launch<empty_kernel_1>(my_config);

  //CHECK: {{define.*kernel.*empty_kernel_1.* !intel_reqd_sub_group_size !}}
  return 0;
}

// Negative test for previous test
int test_no_kernel_properties() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  compat_exp::launch_policy my_config(sycl::nd_range<1>{{32}, {32}});
  compat_exp::launch<empty_kernel_2>(my_config);

  //CHECK-NOT: {{define.*kernel.*empty_kernel_2.* !intel_reqd_sub_group_size !}}
  return 0;
}
