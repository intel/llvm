//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/kernel_name_str_t.hpp>

#include <ur_api.h>

#include <memory>
#include <tuple>
#include <variant>
#include <vector>

namespace sycl {
inline namespace _V1 {
class event;

namespace detail {
class CGExecKernel;
class queue_impl;
class RTDeviceBinaryImage;

std::tuple<const RTDeviceBinaryImage *, ur_program_handle_t>
retrieveKernelBinary(queue_impl &Queue, KernelNameStrRefT KernelName,
                     CGExecKernel *CGKernel = nullptr);

template <typename DereferenceImpl, typename... Iterators>
class variadic_iterator {
  using storage_iter = std::variant<Iterators...>;

  storage_iter It;

public:
  template <typename IterTy>
  variadic_iterator(IterTy &&It) : It(std::forward<IterTy>(It)) {}

  variadic_iterator &operator++() {
    It = std::visit(
        [](auto &&It) {
          ++It;
          return storage_iter{It};
        },
        It);
    return *this;
  }
  bool operator!=(const variadic_iterator &Other) const {
    return It != Other.It;
  }

  decltype(auto) operator*() {
    return std::visit(
        [](auto &&It) -> decltype(auto) {
          return DereferenceImpl::dereference(*It);
        },
        It);
  }
};

// Non-owning!
template <typename iterator> class iterator_range {
public:
  iterator_range(const iterator_range &Other) = default;

  template <typename IterTy>
  iterator_range(IterTy Begin, IterTy End, size_t Size)
      : Begin(Begin), End(End), Size(Size) {}

  template <typename ContainerTy>
  iterator_range(const ContainerTy &Container)
      : iterator_range(Container.begin(), Container.end(), Container.size()) {}

  iterator begin() const { return Begin; }
  iterator end() const { return End; }
  size_t size() const { return Size; }
  bool empty() const { return Size == 0; }
  decltype(auto) front() const { return *begin(); }

private:
  iterator Begin;
  iterator End;
  const size_t Size;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
