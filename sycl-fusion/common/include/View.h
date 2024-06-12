//==------------- View.h - Non-owning view to contiguous memory ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_VIEW_H
#define SYCL_FUSION_COMMON_VIEW_H

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace jit_compiler {

/// Read-only, non-owning view of a linear sequence of \p T.
template <typename T> class View {
public:
  constexpr View(const T *Ptr, size_t Size) : Ptr(Ptr), Size(Size) {}

  template <typename C, typename = std::enable_if_t<
                            std::is_same_v<T, typename C::value_type>>>
  constexpr View(const C &Cont) : Ptr(std::data(Cont)), Size(std::size(Cont)) {}

  constexpr const T *begin() const { return Ptr; }
  constexpr const T *end() const { return Ptr + Size; }
  constexpr size_t size() const { return Size; }

  template <template <typename...> typename C> auto to() const {
    return C<T>{begin(), end()};
  }

private:
  const T *const Ptr;
  const size_t Size;
};

} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_VIEW_H
