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

// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK1 2>&1 | FileCheck -vv %s --check-prefixes=CHECK1
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK2 2>&1 | FileCheck -vv %s --check-prefixes=CHECK2
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK3 2>&1 | FileCheck -vv %s --check-prefixes=CHECK3
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK4 2>&1 | FileCheck -vv %s --check-prefixes=CHECK4
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK5 2>&1 | FileCheck -vv %s --check-prefixes=CHECK5
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK6 2>&1 | FileCheck -vv %s --check-prefixes=CHECK6
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK7 2>&1 | FileCheck -vv %s --check-prefixes=CHECK7
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK8 2>&1 | FileCheck -vv %s --check-prefixes=CHECK8
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK9 2>&1 | FileCheck -vv %s --check-prefixes=CHECK9
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK10 2>&1 | FileCheck -vv %s --check-prefixes=CHECK10
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK11 2>&1 | FileCheck -vv %s --check-prefixes=CHECK11
// RUN: not %clangxx -fsycl -fsyntax-only %s -DCHECK12 2>&1 | FileCheck -vv %s --check-prefixes=CHECK12

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

#ifdef CHECK1
  // Missing range
  {
    compat_exp::launch_policy missing_range_config(
        compat_exp::kernel_properties{sycl_exp::sub_group_size<32>});
    //CHECK1: error: static assertion failed due to requirement 'syclcompat::detail::is_range_or_nd_range_v
  }
#endif
#ifdef CHECK2
  // Duplicate nd_range
  {
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy duplicate_nd_range_config(launch_range, launch_range);
    //CHECK2: error: static assertion failed{{.*Did you forget to wrap}}
  }
#endif
#ifdef CHECK3
  // Duplicate range
  {
    sycl::range<3> launch_range{1,1,32};
    compat_exp::launch_policy duplicate_nd_range_config(launch_range, launch_range);
    //CHECK3: error: static assertion failed{{.*Did you forget to wrap}}
  }
#endif
#ifdef CHECK4
  // Unwrapped property
  {
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy unwrapped_property_config(launch_range, {sycl_exp::sub_group_size<32>});
    //CHECK4: error: no viable constructor or deduction guide for deduction of template arguments of 'compat_exp::launch_policy'
  }
#endif
#ifdef CHECK5
  // Foreign object in ctor
  {
    dummy_properties foreign_object{sycl_exp::sub_group_size<32>};
    sycl::nd_range<3> launch_range{{1,1,32},{1,1,32}};
    compat_exp::launch_policy unwrapped_property_config(launch_range, foreign_object);
    //CHECK5: error: static assertion failed{{.*Did you forget to wrap}}
  }
#endif
#ifdef CHECK6
  // Local mem with sycl::range launch 1
  {
    sycl::range<3> launch_range{1, 1, 32};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy range_and_local_mem_config_1(launch_range,
                                                         lmem_size);
    //CHECK6: error: static assertion failed due to requirement 'syclcompat::detail::is_nd_range_v<sycl::range<3>> || !true': sycl::range kernel launches are incompatible with local
  }
#endif
#ifdef CHECK7
  // Local mem with sycl::range launch 2
  {
    syclcompat::dim3 launch_range{32, 1, 1};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy range_and_local_mem_config_2(launch_range, compat_exp::kernel_properties{sycl_exp::sub_group_size<32>},
                                                         lmem_size);
    //CHECK7: error: static assertion failed due to requirement 'syclcompat::detail::is_nd_range_v<sycl::range<3>> || !true': sycl::range kernel launches are incompatible with local
  }
#endif
#ifdef CHECK8
  // Duplicate local_mem spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy duplicate_local_mem_config(launch_range, lmem_size, lmem_size);
    //CHECK8: error: static assertion failed due to requirement{{.*(exactly once|duplicate type)}}
  }
#endif
#ifdef CHECK9
  // Duplicate kernel_properties spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::kernel_properties kernel_props{sycl_exp::sub_group_size<32>};
    compat_exp::launch_policy duplicate_kernel_properties_config(launch_range, kernel_props, kernel_props);
    //CHECK9: error: static assertion failed due to requirement{{.*type appears more than once}}
  }
#endif
#ifdef CHECK10
  // Duplicate launch_properties spec
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::launch_properties launch_props{};
    compat_exp::local_mem_size lmem_size(0);
    compat_exp::launch_policy duplicate_launch_properties_config(launch_range, launch_props, lmem_size, launch_props);
    //CHECK10: error: static assertion failed due to requirement{{.*type appears more than once}}
  }
#endif
#ifdef CHECK11
  // Missing kernel args
  {
    sycl::range<3> launch_range{1, 1, 32};
    compat_exp::launch_policy range_only(launch_range);
    compat_exp::launch<int_kernel>(range_only);
    //CHECK11: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }
#endif
#ifdef CHECK12
  // Extra kernel args
  {
    sycl::nd_range<3> launch_range{{1, 1, 32}, {1, 1, 32}};
    compat_exp::launch_policy range_only(launch_range);
    int extra_arg = 1;
    compat_exp::launch<empty_kernel>(range_only, extra_arg);
    //CHECK12: error: static assertion failed due to requirement 'syclcompat::args_compatible
  }
#endif
}
