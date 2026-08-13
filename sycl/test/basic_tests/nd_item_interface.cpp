// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Wno-deprecated-declarations -fsyntax-only %s

//==--------- nd_item_interface.cpp - nd_item interface test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/multi_ptr.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/vector.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

template <typename T, typename = void>
struct has_member_equal : std::false_type {};

template <typename T>
struct has_member_equal<
    T, std::void_t<decltype(std::declval<const T &>().operator==(
           std::declval<const T &>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_member_not_equal : std::false_type {};

template <typename T>
struct has_member_not_equal<
    T, std::void_t<decltype(std::declval<const T &>().operator!=(
           std::declval<const T &>()))>> : std::true_type {};

using Item = sycl::nd_item<1>;

static_assert(!std::is_default_constructible_v<Item>);

static_assert(noexcept(std::declval<const Item &>().get_global_id()));
static_assert(noexcept(std::declval<const Item &>().get_global_id(0)));
static_assert(noexcept(std::declval<const Item &>().get_global_linear_id()));
static_assert(noexcept(std::declval<const Item &>().get_local_id()));
static_assert(noexcept(std::declval<const Item &>().get_local_id(0)));
static_assert(noexcept(std::declval<const Item &>().get_local_linear_id()));
static_assert(noexcept(std::declval<const Item &>().get_group()));
static_assert(noexcept(std::declval<const Item &>().get_sub_group()));
static_assert(noexcept(std::declval<const Item &>().get_group(0)));
static_assert(noexcept(std::declval<const Item &>().get_group_linear_id()));
static_assert(noexcept(std::declval<const Item &>().get_group_range()));
static_assert(noexcept(std::declval<const Item &>().get_group_range(0)));
static_assert(noexcept(std::declval<const Item &>().get_global_range()));
static_assert(noexcept(std::declval<const Item &>().get_global_range(0)));
static_assert(noexcept(std::declval<const Item &>().get_local_range()));
static_assert(noexcept(std::declval<const Item &>().get_local_range(0)));
static_assert(noexcept(std::declval<const Item &>().get_offset()));
static_assert(noexcept(std::declval<const Item &>().get_nd_range()));

static_assert(!has_member_equal<Item>::value);
static_assert(!has_member_not_equal<Item>::value);
static_assert(std::is_same_v<decltype(std::declval<const Item &>() ==
                                      std::declval<const Item &>()),
                             bool>);
static_assert(std::is_same_v<decltype(std::declval<const Item &>() !=
                                      std::declval<const Item &>()),
                             bool>);
static_assert(noexcept(std::declval<const Item &>() ==
                       std::declval<const Item &>()));
static_assert(noexcept(std::declval<const Item &>() !=
                       std::declval<const Item &>()));
static_assert(noexcept(operator==(std::declval<const Item &>(),
                                  std::declval<const Item &>())));
static_assert(noexcept(operator!=(std::declval<const Item &>(),
                                  std::declval<const Item &>())));

using LegacyLocalIntPtr = sycl::local_ptr<int>;
using LegacyGlobalIntPtr = sycl::global_ptr<int>;
using DecoratedLocalIntPtr = sycl::decorated_local_ptr<int>;
using DecoratedGlobalIntPtr = sycl::decorated_global_ptr<int>;
using DecoratedLocalConstIntPtr = sycl::decorated_local_ptr<const int>;
using DecoratedGlobalConstIntPtr = sycl::decorated_global_ptr<const int>;

static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyLocalIntPtr>(), std::declval<LegacyGlobalIntPtr>(),
    std::size_t{}, std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyGlobalIntPtr>(), std::declval<LegacyLocalIntPtr>(),
    std::size_t{}, std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedLocalIntPtr>(),
    std::declval<DecoratedGlobalConstIntPtr>(), std::size_t{}, std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedGlobalIntPtr>(),
    std::declval<DecoratedLocalConstIntPtr>(), std::size_t{}, std::size_t{})));

using BoolVector = sycl::vec<bool, 2>;
using LegacyLocalBoolPtr = sycl::local_ptr<bool>;
using LegacyGlobalBoolPtr = sycl::global_ptr<bool>;
using LegacyLocalBoolVectorPtr = sycl::local_ptr<BoolVector>;
using LegacyGlobalBoolVectorPtr = sycl::global_ptr<BoolVector>;
using DecoratedLocalBoolPtr = sycl::decorated_local_ptr<bool>;
using DecoratedGlobalConstBoolPtr = sycl::decorated_global_ptr<const bool>;
using DecoratedLocalBoolVectorPtr = sycl::decorated_local_ptr<BoolVector>;
using DecoratedGlobalConstBoolVectorPtr =
    sycl::decorated_global_ptr<const BoolVector>;

static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyLocalBoolPtr>(), std::declval<LegacyGlobalBoolPtr>(),
    std::size_t{}, std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyLocalBoolVectorPtr>(),
    std::declval<LegacyGlobalBoolVectorPtr>(), std::size_t{}, std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedLocalBoolPtr>(),
    std::declval<DecoratedGlobalConstBoolPtr>(), std::size_t{},
    std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedLocalBoolVectorPtr>(),
    std::declval<DecoratedGlobalConstBoolVectorPtr>(), std::size_t{},
    std::size_t{})));

static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyLocalIntPtr>(), std::declval<LegacyGlobalIntPtr>(),
    std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<LegacyGlobalIntPtr>(), std::declval<LegacyLocalIntPtr>(),
    std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedLocalIntPtr>(),
    std::declval<DecoratedGlobalConstIntPtr>(), std::size_t{})));
static_assert(noexcept(std::declval<const Item &>().async_work_group_copy(
    std::declval<DecoratedGlobalIntPtr>(),
    std::declval<DecoratedLocalConstIntPtr>(), std::size_t{})));

static_assert(noexcept(
    std::declval<const Item &>().wait_for(std::declval<sycl::device_event>())));
