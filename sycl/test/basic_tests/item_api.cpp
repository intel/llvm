// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Wno-deprecated-declarations -fsyntax-only %s
//==----------- item_api.cpp - SYCL item API test --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/item.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

template <typename T, typename = void>
struct HasMemberEquality : std::false_type {};

template <typename T>
struct HasMemberEquality<T, std::void_t<decltype(&T::operator==)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasMemberInequality : std::false_type {};

template <typename T>
struct HasMemberInequality<T, std::void_t<decltype(&T::operator!=)>>
    : std::true_type {};

using ItemWithOffset = sycl::item<2, true>;
using ItemWithoutOffset = sycl::item<2, false>;
using OneDimItem = sycl::item<1, false>;

static_assert(noexcept(std::declval<const ItemWithOffset &>().get_id()));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_id(0)));
static_assert(noexcept(std::declval<const ItemWithOffset &>()[0]));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_range()));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_range(0)));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_offset()));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_offset(0)));
static_assert(noexcept(std::declval<const ItemWithOffset &>().get_linear_id()));
static_assert(
    noexcept(static_cast<std::size_t>(std::declval<const OneDimItem &>())));
static_assert(noexcept(static_cast<sycl::item<2, true>>(
    std::declval<const ItemWithoutOffset &>())));
static_assert(noexcept(std::declval<const ItemWithOffset &>() ==
                       std::declval<const ItemWithOffset &>()));
static_assert(noexcept(std::declval<const ItemWithOffset &>() !=
                       std::declval<const ItemWithOffset &>()));

static_assert(!HasMemberEquality<ItemWithOffset>::value);
static_assert(!HasMemberInequality<ItemWithOffset>::value);
static_assert(noexcept(operator==(std::declval<const ItemWithOffset &>(),
                                  std::declval<const ItemWithOffset &>())));
static_assert(noexcept(operator!=(std::declval<const ItemWithOffset &>(),
                                  std::declval<const ItemWithOffset &>())));
