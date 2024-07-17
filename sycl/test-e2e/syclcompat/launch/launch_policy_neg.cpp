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
 *  launch_policy_neg.cpp
 *
 *  Description:
 *     Negative tests for new launch_policy.
 **************************************************************************/

// RUN: not %clangxx -std=c++20 -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out 2>&1 | FileCheck -vv %s

#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/group_barrier.hpp>

#include <syclcompat/launch.hpp>
#include <syclcompat/memory.hpp>
#include <syclcompat/dims.hpp>

namespace compat_exp = syclcompat::experimental;
namespace sycl_exp = sycl::ext::oneapi::experimental;

// Notes on use of FileCheck here:
// Failures do not necessarily occur in order (hence use of CHECK-DAG)
// Additionally a `static_assert` hit during a template instantiation will only
// be hit once per unique concrete class. The only solution (aside from hacking
// the examples to have different template types) would presumably be multiple
// compilation units?

// Dummy kernels for testing
inline void empty_kernel(){};
inline void int_kernel(int a){};
inline void int_ptr_kernel(int *a){};

inline void dynamic_local_mem_empty_kernel(char *a){};

template <typename T>
inline void dynamic_local_mem_basicdt_kernel(T value, char *local_mem){};


// Dummy property container for negative testing
template <typename Properties> struct dummy_properties {
  static_assert(sycl_exp::is_property_list_v<Properties>);
  using Props = Properties;

  template <typename... Props>
  dummy_properties(Props... properties) : props{properties...} {}

  Properties props;
};
template <typename... Props>
dummy_properties(Props... props)
    -> dummy_properties<decltype(sycl_exp::properties(props...))>;

void test_variadic_config_ctor() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  // Missing range
  {
    compat_exp::launch_policy missing_range_config(
        compat_exp::kernel_properties{sycl_exp::sub_group_size<32>});
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::detail::is_range_or_nd_range_v
  }

  // Duplicate nd_range
  {
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy duplicate_nd_range_config(launch_range, launch_range);
    //CHECK-DAG: error: static assertion failed due to requirement 'std::conjunction_v<std::disjunction<
  }

  // Duplicate range
  {
    sycl::range<3> launch_range{1,1,32};
    compat_exp::launch_policy duplicate_nd_range_config(launch_range, launch_range);
    //CHECK-DAG: error: static assertion failed due to requirement 'std::conjunction_v<std::disjunction<
  }

  // Unwrapped property
  {
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy unwrapped_property_config(launch_range, {sycl_exp::sub_group_size<32>});
    //CHECK-DAG: error: no viable constructor or deduction guide for deduction of template arguments of 'compat_exp::launch_policy'
  }

  // Foreign object in ctor
  {
    dummy_properties foreign_object{sycl_exp::sub_group_size<32>};
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy unwrapped_property_config(launch_range, foreign_object);
    //CHECK-DAG: error: no viable constructor or deduction guide for deduction of template arguments of 'compat_exp::launch_policy'
  }
  // Local mem with sycl::range launch 1
  {
    sycl::range<3> launch_range{1, 1, 32};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy range_and_local_mem_config_1(launch_range,
                                                         lmem_size);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::detail::is_nd_range_v<sycl::range<3>> || !true': sycl::range kernel launches are incompatible with local
  }
  // Local mem with sycl::range launch 2
  {
    syclcompat::dim3 launch_range{32, 1, 1};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy range_and_local_mem_config_2(launch_range, compat_exp::kernel_properties{sycl_exp::sub_group_size<32>},
                                                         lmem_size);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::detail::is_nd_range_v<sycl::range<3>> || !true': sycl::range kernel launches are incompatible with local
  }
  // Duplicate local_mem spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy duplicate_local_mem_config(launch_range, lmem_size, lmem_size);
    //CHECK-DAG: error: static assertion failed due to requirement{{.*exactly once}}
  }

  // Duplicate kernel_properties spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::kernel_properties kernel_props{sycl_exp::sub_group_size<32>};
    compat_exp::launch_policy duplicate_kernel_properties_config(launch_range, kernel_props, kernel_props);
    //CHECK-DAG: error: static assertion failed due to requirement{{.*type appears more than once}}
  }

  // Duplicate launch_properties spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::launch_properties launch_props{};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy duplicate_launch_properties_config(launch_range, launch_props, lmem_size, launch_props);
    //CHECK-DAG: error: static assertion failed due to requirement{{.*type appears more than once}}
  }

  // Missing kernel args
  {
    sycl::range<3> launch_range{1, 1, 32};
    compat_exp::launch_policy range_only(launch_range);
    compat_exp::launch<int_kernel>(range_only);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }

  // Extra kernel args
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::launch_policy range_only(launch_range);
    int extra_arg = 1;
    compat_exp::launch<empty_kernel>(range_only, extra_arg);
    //CHECK-DAG: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }

}
