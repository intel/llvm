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

using value_type_list =
    std::tuple<int, unsigned int, short, unsigned short, long, unsigned long,
               long long, unsigned long long, float, double, sycl::half>;

using fp_type_list = std::tuple<float, double, sycl::half>;
