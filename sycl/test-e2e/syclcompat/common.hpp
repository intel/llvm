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

// Typed call helper
// Iterates over all types and calls Functor f for each of them
template <typename types, typename Functor>
void instantiate_all_types(Functor &&f) {
  auto for_each_type_call =
      [&]<template <typename...> class T, typename... Rest>(T<Rest...> *) {
        (f.template operator()<Rest>(), ...);
      };
  for_each_type_call(static_cast<types *>(nullptr));
}

#define INSTANTIATE_ALL_TYPES(types, f)                                        \
  instantiate_all_types<types>([]<typename T>() { f<T>(); });
