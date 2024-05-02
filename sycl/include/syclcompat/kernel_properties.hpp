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
 *  kernel_properties.hpp
 *
 *  Description:
 *    Provides utility structs for commonly used kernel attributes, like
 *    reqd_sub_group_size, maximum work-items per work-group, minimum
 *    work groups per compute unit and maximum cluster size.
 *    Also provides quick utility structs using subgorup size 16 and 8
 *    Utilizes the following extension - 
 *      sycl_ext_oneapi_kernel_properties
 **************************************************************************/

#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <sycl/sycl.hpp>

#if defined(SYCL_EXT_ONEAPI_KERNEL_PROPERTIES)

namespace syclcompat {
namespace experimental {

constexpr auto empty_property_list =
    sycl::ext::oneapi::experimental::properties{};

struct EmptyKernelPropertyStruct {
  static constexpr auto kernel_properties = empty_property_list;
};

template <int SubgroupSize> struct ReqdSubGroupSizeStruct {
  static constexpr auto kernel_properties =
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
};

struct ReqdSubGroupSize16 : ReqdSubGroupSizeStruct<16>;
struct ReqdSubGroupSize8 : ReqdSubGroupSizeStruct<8>;

} // namespace experimental
} // namespace syclcompat

#endif
