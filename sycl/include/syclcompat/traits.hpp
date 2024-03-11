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
 *  traits.hpp
 *
 *  Description:
 *    Type traits for the SYCL compatibility extension
 **************************************************************************/

#pragma once

#include <cstddef>
#include <type_traits>

namespace syclcompat {

// Equivalent to C++20's std::type_identity (used to create non-deduced
// contexts)
template <class T> struct type_identity {
  using type = T;
};
template <class T> using type_identity_t = typename type_identity<T>::type;

// Defines the operand type for arithemtic operations on T. This is identity
// for all types except pointers, for which it is std::ptrdiff_t
template <typename T> struct arith {
  using type = std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
};
template <typename T> using arith_t = typename arith<T>::type;

} // namespace syclcompat
