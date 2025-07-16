//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/impl_utils.hpp>
#include <sycl/detail/kernel_name_str_t.hpp>
#include <sycl/detail/type_traits.hpp>

#include <ur_api.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <queue>
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

const RTDeviceBinaryImage *
retrieveKernelBinary(queue_impl &Queue, KernelNameStrRefT KernelName,
                     CGExecKernel *CGKernel = nullptr);

template <typename DereferenceImpl, typename SyclTy, typename... Iterators>
class variadic_iterator {
  using storage_iter = std::variant<Iterators...>;

  storage_iter It;

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using reference = decltype(DereferenceImpl::dereference(
      *std::declval<nth_type_t<0, Iterators...>>()));
  using value_type = std::remove_reference_t<reference>;
  using sycl_type = SyclTy;
  using pointer = value_type *;
  static_assert(std::is_same_v<reference, value_type &>);

  variadic_iterator(const variadic_iterator &) = default;
  variadic_iterator(variadic_iterator &&) = default;
  variadic_iterator(variadic_iterator &) = default;
  variadic_iterator& operator=(const variadic_iterator &) = default;
  variadic_iterator& operator=(variadic_iterator &&) = default;

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
  bool operator==(const variadic_iterator &Other) const {
    return It == Other.It;
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
  using value_type = typename iterator::value_type;
  using sycl_type = typename iterator::sycl_type;

  template <typename Container, typename = void>
  struct has_reserve : public std::false_type {};
  template <typename Container>
  struct has_reserve<
      Container, std::void_t<decltype(std::declval<Container>().reserve(1))>>
      : public std::true_type {};

public:
  iterator_range(const iterator_range &Other) = default;

  template <typename IterTy>
  iterator_range(IterTy Begin, IterTy End, size_t Size)
      : Begin(Begin), End(End), Size(Size) {}

  iterator_range()
      : iterator_range(static_cast<value_type *>(nullptr),
                       static_cast<value_type *>(nullptr), 0) {}

  template <typename ContainerTy>
  iterator_range(const ContainerTy &Container)
      : iterator_range(Container.begin(), Container.end(), Container.size()) {}

  iterator_range(value_type &Obj) : iterator_range(&Obj, &Obj + 1, 1) {}

  iterator_range(const sycl_type &Obj)
      : iterator_range(&*getSyclObjImpl(Obj), (&*getSyclObjImpl(Obj) + 1), 1) {}

  iterator begin() const { return Begin; }
  iterator end() const { return End; }
  size_t size() const { return Size; }
  bool empty() const { return Size == 0; }
  decltype(auto) front() const { return *begin(); }

  template <typename Container>
  std::enable_if_t<
      check_type_in_v<Container, std::vector<sycl_type>,
                      std::queue<value_type *>, std::vector<value_type *>,
                      std::vector<std::shared_ptr<value_type>>>,
      Container>
  to() const {
    std::conditional_t<std::is_same_v<Container, std::queue<value_type *>>,
                       typename std::queue<value_type *>::container_type,
                       Container>
        Result;
    if constexpr (has_reserve<decltype(Result)>::value)
      Result.reserve(size());
    std::transform(
        begin(), end(), std::back_inserter(Result), [](value_type &E) {
          if constexpr (std::is_same_v<Container, std::vector<sycl_type>>)
            return createSyclObjFromImpl<sycl_type>(E);
          else if constexpr (std::is_same_v<
                                 Container,
                                 std::vector<std::shared_ptr<value_type>>>)
            return E.shared_from_this();
          else
            return &E;
        });
    if constexpr (std::is_same_v<Container, decltype(Result)>)
      return Result;
    else
      return Container{std::move(Result)};
  }

  bool contains(value_type &Other) const {
    return std::find_if(begin(), end(), [&Other](value_type &Elem) {
             return &Elem == &Other;
           }) != end();
  }

protected:
  template <typename Container>
  static constexpr bool has_reserve_v = has_reserve<Container>::value;

private:
  iterator Begin;
  iterator End;
  const size_t Size;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
