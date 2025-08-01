// RUN: %clangxx -fsycl -fsyntax-only %s

#include <type_traits>

#include <cstdlib>

// Unqualified calls should resolve to the `int` overloads
static_assert(std::is_same_v<decltype(div(1, 1)), div_t>);
static_assert(std::is_same_v<decltype(div(1l, 1l)), div_t>);
static_assert(std::is_same_v<decltype(div(1ll, 1ll)), div_t>);
static_assert(std::is_same_v<decltype(abs(1)), int>);
static_assert(std::is_same_v<decltype(abs(1l)), int>);
static_assert(std::is_same_v<decltype(abs(1ll)), int>);
