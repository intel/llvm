//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/impl_utils.hpp>
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
retrieveKernelBinary(queue_impl &Queue, std::string_view KernelName,
                     CGExecKernel *CGKernel = nullptr);

template <typename SyclTy, typename... Iterators> class variadic_iterator {
  using storage_iter = std::variant<Iterators...>;

  storage_iter It;

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_reference_t<decltype(*getSyclObjImpl(
      std::declval<SyclTy>()))>;
  using sycl_type = SyclTy;
  using pointer = value_type *;
  using reference = value_type &;

  variadic_iterator() = default;
  variadic_iterator(const variadic_iterator &) = default;
  variadic_iterator(variadic_iterator &&) = default;
  variadic_iterator(variadic_iterator &) = default;
  variadic_iterator &operator=(const variadic_iterator &) = default;
  variadic_iterator &operator=(variadic_iterator &&) = default;

  template <typename IterTy>
  variadic_iterator(IterTy &&It) : It(std::forward<IterTy>(It)) {}

  variadic_iterator &operator++() noexcept {
    std::visit([](auto &&It) noexcept { ++It; }, It);
    return *this;
  }
  bool operator!=(const variadic_iterator &Other) const noexcept {
    return It != Other.It;
  }
  bool operator==(const variadic_iterator &Other) const noexcept {
    return It == Other.It;
  }

  decltype(auto) operator*() noexcept {
    return std::visit(
        [](auto &&It) -> decltype(auto) {
          decltype(auto) Elem = *It;
          using Ty = std::decay_t<decltype(Elem)>;
          static_assert(!std::is_same_v<Ty, decltype(Elem)>);
          if constexpr (std::is_same_v<Ty, sycl_type>) {
            return *getSyclObjImpl(Elem);
          } else if constexpr (std::is_same_v<Ty, value_type>) {
            return Elem;
          } else {
            return *Elem;
          }
        },
        It);
  }

  pointer operator->() { return &this->operator*(); }
};

// Non-owning!
template <typename iterator> class iterator_range {
  using value_type = typename iterator::value_type;

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

  iterator_range() : iterator_range(iterator{}, iterator{}, 0) {}

  template <typename ContainerTy, typename = std::void_t<decltype(iterator{
                                      std::declval<ContainerTy>().begin()})>>
  iterator_range(const ContainerTy &Container)
      : iterator_range(Container.begin(), Container.end(), Container.size()) {}

  iterator_range(value_type &Obj) : iterator_range(&Obj, &Obj + 1, 1) {}

  template <typename sycl_type,
            typename = std::void_t<decltype(iterator{
                &*getSyclObjImpl(std::declval<sycl_type>())})>,
            // To make it different from `ContainerTy` overload above:
            typename = void>
  iterator_range(const sycl_type &Obj)
      : iterator_range(&*getSyclObjImpl(Obj), (&*getSyclObjImpl(Obj) + 1), 1) {}

  iterator begin() const { return Begin; }
  iterator end() const { return End; }
  size_t size() const { return Size; }
  bool empty() const { return Size == 0; }
  decltype(auto) front() const { return *begin(); }

  // Only enable for ranges of `variadic_iterator` and for the containers with
  // proper `value_type`. The last part is important so that descendent
  // `devices_range` could provide its own specialization for
  // `to<std::vector<device_handle_t>>()`.
  template <typename Container, typename iterator_ = iterator,
            typename = std::enable_if_t<check_type_in_v<
                typename Container::value_type, value_type *,
                std::shared_ptr<value_type>, typename iterator_::sycl_type>>>
  Container to() const {
    std::conditional_t<std::is_same_v<Container, std::queue<value_type *>>,
                       typename std::queue<value_type *>::container_type,
                       Container>
        Result;
    if constexpr (has_reserve<decltype(Result)>::value)
      Result.reserve(size());
    std::transform(
        begin(), end(), std::back_inserter(Result), [](value_type &E) {
          using container_value_type = typename Container::value_type;
          if constexpr (std::is_same_v<container_value_type,
                                       std::shared_ptr<value_type>>)
            return E.shared_from_this();
          else if constexpr (std::is_same_v<container_value_type, value_type *>)
            return &E;
          else
            return createSyclObjFromImpl<container_value_type>(E);
        });
    if constexpr (std::is_same_v<Container, decltype(Result)>)
      return Result;
    else
      return Container{std::move(Result)};
  }

  // Only enable for ranges of `variadic_iterator` above.
  template <typename T = iterator,
            typename = std::void_t<typename T::sycl_type>>
  bool contains(value_type &Other) const {
    return std::find_if(begin(), end(), [&Other](value_type &Elem) {
             return &Elem == &Other;
           }) != end();
  }

private:
  iterator Begin;
  iterator End;
  const size_t Size;

  template <class Pred> friend bool all_of(iterator_range R, Pred &&P) {
    return std::all_of(R.begin(), R.end(), std::forward<Pred>(P));
  }

  template <class Pred> friend bool any_of(iterator_range R, Pred &&P) {
    return std::any_of(R.begin(), R.end(), std::forward<Pred>(P));
  }

  template <class Pred> friend bool none_of(iterator_range R, Pred &&P) {
    return std::none_of(R.begin(), R.end(), std::forward<Pred>(P));
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl
