// RUN: %clangxx -fsycl -fsyntax-only %s

#include <type_traits>

#include <cstdlib>

int i = 1;
long l = 1;
long long ll = 1;

// Unqualified calls should resolve to the `int` overloads
static_assert(std::is_same_v<decltype(abs(i)), int>);
static_assert(std::is_same_v<decltype(div(i, i)), div_t>);
// NOTE: Windows Universal C Runtime defines C++ overloads for `long` and
// `long long` types in global name space. See
// https://github.com/huangqinjin/ucrt/blob/d6e817a4cc90f6f1fe54f8a0aa4af4fff0bb647d/include/stdlib.h#L360-L383.
#ifndef _WIN32
static_assert(std::is_same_v<decltype(abs(l)), int>);
static_assert(std::is_same_v<decltype(abs(ll)), int>);
static_assert(std::is_same_v<decltype(div(i, l)), div_t>);
static_assert(std::is_same_v<decltype(div(i, ll)), div_t>);
static_assert(std::is_same_v<decltype(div(l, i)), div_t>);
static_assert(std::is_same_v<decltype(div(l, l)), div_t>);
static_assert(std::is_same_v<decltype(div(l, ll)), div_t>);
static_assert(std::is_same_v<decltype(div(ll, i)), div_t>);
static_assert(std::is_same_v<decltype(div(ll, l)), div_t>);
static_assert(std::is_same_v<decltype(div(ll, ll)), div_t>);
#endif // _WIN32
