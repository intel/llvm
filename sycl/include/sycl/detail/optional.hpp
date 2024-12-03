//==-------- optional.hpp - limited variant of std::optional -------- C++ --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <optional>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

// ABI-stable implementation of optional to avoid reliance on potentially
// differing implementations of std::optional when crossing the library
// boundary.
template <typename T> class optional {
public:
  constexpr optional() noexcept {}
  constexpr optional(std::nullopt_t) noexcept : optional() {}

  template <typename U>
  constexpr optional(const optional<U> &Other)
      : ContainsValue{Other.ContainsValue} {
    new (Storage) T(Other.Value);
  }
  template <typename U>
  constexpr optional(optional<U> &&Other)
      : ContainsValue{std::move(Other.ContainsValue)} {
    new (Storage) T(std::move(Other.Value));
  }

  constexpr optional(T &&Value) : ContainsValue{true} {
    new (Storage) T(std::move(Value));
  }

  constexpr optional(const T &Value) : ContainsValue{true} {
    new (Storage) T(Value);
  }

  template <typename U>
  constexpr optional(const std::optional<U> &Other) : ContainsValue{Other} {
    if (Other)
      new (Storage) T(*Other);
  }

  ~optional() {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
  }

  optional &operator=(std::nullopt_t) noexcept {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = false;
    return *this;
  }

  template <typename U> optional &operator=(const optional<U> &Other) {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = Other;
    new (Storage) T(Other.Value);
    return *this;
  }
  template <typename U> optional &operator=(optional<U> &&Other) noexcept {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = Other;
    new (Storage) T(std::move(Other.Value));
    return *this;
  }

  optional &operator=(T &&Value) {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = true;
    new (Storage) T(std::move(Value));
    return *this;
  }

  optional &operator=(const T &Value) {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = true;
    new (Storage) T(Value);
    return *this;
  }

  template <typename U> optional &operator=(const std::optional<U> &Other) {
    if (has_value())
      reinterpret_cast<T *>(Storage)->~T();
    ContainsValue = Other;
    if (Other)
      new (Storage) T(*Other);
    return *this;
  }

  constexpr bool has_value() const noexcept { return ContainsValue; }
  constexpr explicit operator bool() const noexcept { return has_value(); }

  constexpr T &value() & {
    if (!has_value())
      throw std::bad_optional_access{};
    return *reinterpret_cast<T *>(Storage);
  }
  constexpr const T &value() const & {
    if (!has_value())
      throw std::bad_optional_access{};
    return *reinterpret_cast<const T *>(Storage);
  }
  constexpr T &&value() && {
    if (!has_value())
      throw std::bad_optional_access{};
    return std::move(*reinterpret_cast<T *>(Storage));
  }
  constexpr const T &&value() const && {
    if (!has_value())
      throw std::bad_optional_access{};
    return std::move(*reinterpret_cast<const T *>(Storage));
  }

  template <class U> constexpr T value_or(U &&DefaultVal) {
    return has_value() ? value() : static_cast<T>(std::forward<U>(DefaultVal));
  }
  template <class U> constexpr T value_or(U &&DefaultVal) const {
    return has_value() ? std::move(value())
                       : static_cast<T>(std::forward<U>(DefaultVal));
  }

  constexpr T &operator*() & { return value(); }
  constexpr const T &operator*() const & { return value(); }
  constexpr T &&operator*() && { return value(); }
  constexpr const T &&operator*() const && { return value(); }

private:
  alignas(alignof(T)) char Storage[sizeof(T)] = {0};
  bool ContainsValue = false;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
