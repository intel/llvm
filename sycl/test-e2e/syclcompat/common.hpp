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

// Typed call helper
// Iterates over all types and calls Functor f for each of them
template <typename Functor, template <typename...> class Container,
          typename... Ts>
void for_each_type_call(Functor &&f, Container<Ts...> *) {
  (f.template operator()<Ts>(), ...);
}

template <typename tuple, typename Functor>
void instantiate_all_types(Functor &&f) {
  for_each_type_call(f, static_cast<tuple *>(nullptr));
}

#define INSTANTIATE_ALL_TYPES(tuple, f)                                        \
  instantiate_all_types<tuple>([]<typename T>() { f<T>(); });

using value_type_list =
    std::tuple<int, unsigned int, short, unsigned short, long, unsigned long,
               long long, unsigned long long, float, double, sycl::half>;

using fp_type_list = std::tuple<float, double, sycl::half>;
