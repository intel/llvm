// RUN: %clangxx -fsycl -fsyntax-only -std=c++17 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -std=c++17 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -std=c++20 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

//==-- sub_group_interface.cpp - SYCL sub_group interface conformance test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that sycl::sub_group matches the synopsis in SYCL 2020 section 4.9.1.8
// and that it cannot be constructed by an application.

#include <sycl/sub_group.hpp>

#include <cstdint>
#include <type_traits>

using SG = sycl::sub_group;

// Member types and static members.
static_assert(std::is_same_v<SG::id_type, sycl::id<1>>);
static_assert(std::is_same_v<SG::range_type, sycl::range<1>>);
static_assert(std::is_same_v<SG::linear_id_type, std::uint32_t>);
static_assert(SG::dimensions == 1);
static_assert(SG::fence_scope == sycl::memory_scope::sub_group);

// Common by-value semantics, SYCL 2020 section 4.5.3.
static_assert(std::is_copy_constructible_v<SG>);
static_assert(std::is_copy_assignable_v<SG>);
static_assert(std::is_move_constructible_v<SG>);
static_assert(std::is_move_assignable_v<SG>);
static_assert(std::is_destructible_v<SG>);

// Applications must not be able to construct a sub_group. Note that the class
// must not be an aggregate either, otherwise `sycl::sub_group{}` would be
// well-formed aggregate initialization in C++17. Since making the default
// constructor unusable is an API break, this is only enforced in preview mode.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
static_assert(!std::is_default_constructible_v<SG>);
static_assert(!std::is_aggregate_v<SG>);

void cannot_be_constructed() {
  // expected-error@+1 {{call to deleted constructor of 'sycl::sub_group'}}
  sycl::sub_group Default;
  // expected-error@+1 {{call to deleted constructor of 'sycl::sub_group'}}
  auto BracedInit = sycl::sub_group{};
  // expected-error@+1 {{call to deleted constructor of 'sycl::sub_group'}}
  auto ParenInit = sycl::sub_group();
}
#else
// expected-no-diagnostics
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
