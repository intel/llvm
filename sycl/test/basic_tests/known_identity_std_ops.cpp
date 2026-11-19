// The KHR type_traits split header provides known_identity but must not drag in
// <functional>: function objects are recognized through
// sycl::detail::known_op_traits, which sycl/functional.hpp specializes.
// RUN: %clangxx -fsycl -fsycl-host-only -fsyntax-only -DDEPS_ONLY -H %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEPS \
// RUN:     --implicit-check-not='{{[/\\]}}functional{{$}}'
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -DDEPS_ONLY -H %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEPS \
// RUN:     --implicit-check-not='{{[/\\]}}functional{{$}}'
// DEPS: split_headers{{[/\\]}}type_traits.hpp

// Op recognition, with split headers and with sycl.hpp.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DFULL %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -std=c++20 %s

// Op recognition when only a single consumer of known_identity is included:
// each of them must make the sycl/functional.hpp registrations visible on its
// own.
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DCONSUMER=1 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DCONSUMER=2 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DCONSUMER=3 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DCONSUMER=4 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DCONSUMER=5 %s
// expected-no-diagnostics

#if !defined(CONSUMER)
#include <sycl/khr/split_headers/type_traits.hpp>
#endif

#ifndef DEPS_ONLY
#if CONSUMER == 1
#include <sycl/group_algorithm.hpp>
#elif CONSUMER == 2
#include <sycl/reduction.hpp>
#elif CONSUMER == 3
#include <complex> // sycl/stl_wrappers/complex
#elif CONSUMER == 4
#include <sycl/khr/split_headers/reduction.hpp>
#elif CONSUMER == 5
#include <sycl/khr/split_headers/group_algorithms.hpp>
#elif defined(FULL)
#include <sycl/sycl.hpp>
#else
#include <sycl/khr/split_headers/functional.hpp>
#endif

#include <functional>

using namespace sycl;
namespace sd = sycl::detail;

// std:: function objects are recognized as an extension, transparent or with
// the exact accumulator type.
static_assert(has_known_identity_v<std::plus<>, int>);
static_assert(has_known_identity_v<std::plus<int>, int>);
static_assert(!has_known_identity_v<std::plus<long>, int>);
static_assert(has_known_identity_v<std::multiplies<int>, int>);
static_assert(has_known_identity_v<std::bit_or<>, unsigned>);
static_assert(has_known_identity_v<std::bit_and<unsigned>, unsigned>);
static_assert(has_known_identity_v<std::bit_xor<>, int>);
static_assert(known_identity_v<std::logical_and<>, bool> == true);
static_assert(known_identity_v<std::logical_or<bool>, bool> == false);

// SYCL function objects.
static_assert(has_known_identity_v<sycl::plus<>, float>);
static_assert(has_known_identity_v<sycl::minimum<int>, int>);
static_assert(!has_known_identity_v<sycl::minimum<long>, int>);
static_assert(has_known_identity_v<sycl::maximum<>, float>);
static_assert(has_known_identity_v<sycl::logical_or<>, bool>);
static_assert(has_known_identity_v<sycl::logical_and<bool>, bool>);

// Unknown operations.
static_assert(!has_known_identity_v<std::minus<>, int>);
static_assert(!has_known_identity_v<std::divides<int>, int>);

// Deterministic wrappers are recognized as the wrapped operation (one level).
static_assert(has_known_identity_v<
              sd::DeterministicOperatorWrapper<std::plus<int>>, int>);
static_assert(has_known_identity_v<
              sd::DeterministicOperatorWrapper<sycl::maximum<>>, int>);
static_assert(!has_known_identity_v<
              sd::DeterministicOperatorWrapper<std::plus<long>>, int>);
static_assert(
    !has_known_identity_v<sd::DeterministicOperatorWrapper<std::minus<>>, int>);
static_assert(
    !has_known_identity_v<sd::DeterministicOperatorWrapper<
                              sd::DeterministicOperatorWrapper<std::plus<>>>,
                          int>);
#endif // DEPS_ONLY
