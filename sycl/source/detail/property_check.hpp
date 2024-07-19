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
                        const std::set<std::pair<int, bool>> &AllowedProps) {
  std::set<std::pair<int, bool>> Props;
  CreationProps.convertPropertiesToKinds(Props);
  if (!std::includes(AllowedProps.begin(), AllowedProps.end(), Props.begin(),
                     Props.end()))
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "The property list contains property unsupported for "
                          "the current object");
}

} // namespace detail
} // namespace _V1
} // namespace sycl