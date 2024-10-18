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
 *  common.hpp
 *
 *  Description:
 *     Common helpers to help with syclcompat functionality tests
 **************************************************************************/

#pragma once

#include <sycl/feature_test.hpp>
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#include <sycl/half_type.hpp>
#include <tuple>

constexpr double ERROR_TOLERANCE = 1e-5;

template <typename Tuple, typename Func, std::size_t... Is>
void for_each_type_call(Func &&f, std::index_sequence<Is...>) {
  (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template <typename Tuple, typename Func> void instantiate_all_types(Func &&f) {
  for_each_type_call<Tuple>(
      std::forward<Func>(f),
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

#define INSTANTIATE_ALL_TYPES(tuple, f)                                        \
  instantiate_all_types<tuple>([](auto index) {                                \
    using T = std::tuple_element_t<decltype(index)::value, tuple>;             \
    f<T>();                                                                    \
  });

#define INSTANTIATE_ALL_CONTAINER_TYPES(tuple, container, f)                   \
  instantiate_all_types<tuple>([](auto index) {                                \
    using T = std::tuple_element_t<decltype(index)::value, tuple>;             \
    f<container, T>();                                                         \
  });

using value_type_list =
    std::tuple<char, signed char, unsigned char, int, unsigned int, short,
               unsigned short, long, unsigned long, long long,
               unsigned long long, float, double, sycl::half
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
               ,sycl::ext::oneapi::bfloat16
#endif
>;

using fp_type_list_no_bfloat16 = std::tuple<float, double, sycl::half>;

using fp_type_list = std::tuple<float, double, sycl::half

#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
                ,sycl::ext::oneapi::bfloat16
#endif
>;

using marray_type_list =
    std::tuple<char, signed char, short, int, long, long long, unsigned char,
               unsigned short, unsigned int, unsigned long, unsigned long long,
               float, double, sycl::half
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
              , sycl::ext::oneapi::bfloat16
#endif
>;
using vec_type_list = std::tuple<int8_t, int16_t, int32_t, int64_t, uint8_t,
                                 uint16_t, uint32_t, uint64_t, float, double,
                                 sycl::half
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
              , sycl::ext::oneapi::bfloat16
#endif
>;
