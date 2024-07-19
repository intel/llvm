//==----------------- property_check.hpp - SYCL property check utils------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/exception.hpp>
#include <sycl/property_list.hpp>

#include <algorithm> // for std::includes
#include <map>       // for std::multimap

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename... PropType>
std::set<std::pair<int, bool>> GenerateAllowedProps() {
  std::set<std::pair<int, bool>> AllowedProps;
  (AllowedProps.insert(
       {PropType::getKind(),
        std::is_base_of_v<detail::PropertyWithDataBase, PropType>}),
   ...);
  return AllowedProps;
}

void checkPropsAndThrow(const sycl::property_list &CreationProps,
                        const std::set<std::pair<int, bool>> &AllowedProps);

} // namespace detail
} // namespace _V1
} // namespace sycl