//==---------------- accessor_iterator.cpp - SYCL standard source file -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/accessor_iterator.hpp>

#include <ostream>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT std::ostream &operator<<(std::ostream &os,
                                       const accessor_iterator_data &it) {
  os << "accessor_iterator {\n";
  os << "\tMLinearId: " << it.MLinearId << "\n";
  os << "\tMEnd: " << it.MEnd << "\n";
  os << "\tMStaticOffset: " << it.MStaticOffset << "\n";
  os << "\tMPerRowOffset: " << it.MPerRowOffset << "\n";
  os << "\tMPerSliceOffset: " << it.MPerSliceOffset << "\n";
  os << "\tMRowSize: " << it.MRowSize << "\n";
  os << "\tMSliceSize: " << it.MSliceSize << "\n";
  os << "\tMAccessorIsRanged: " << it.MAccessorIsRanged << "\n";
  os << "}";
  return os;
}

} // namespace detail
} // namespace _V1
} // namespace sycl