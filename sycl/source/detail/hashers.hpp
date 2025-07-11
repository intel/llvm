//==--------------------- hashers.hpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include <set>
#include <utility>
#include <vector>
namespace {
// https://stackoverflow.com/questions/35985960
template <class T> inline void hash_combine(std::size_t &seed, const T &v) {
  seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace
namespace std {
// Hash functions used by the kernel/program cache data structures.
template <typename S, typename T> struct hash<pair<S, T>> {
  size_t operator()(const pair<S, T> &v) const {
    size_t seed = 0;
    hash_combine(seed, v.first);
    hash_combine(seed, v.second);
    return seed;
  }
};

template <typename S> struct hash<set<S>> {
  size_t operator()(const set<S> &v) const {
    size_t seed = 0;
    for (const auto &el : v) {
      hash_combine(seed, el);
    }
    return seed;
  }
};

template <typename S> struct hash<vector<S>> {
  size_t operator()(const vector<S> &v) const {
    size_t seed = 0;
    for (const auto &el : v) {
      hash_combine(seed, el);
    }
    return seed;
  }
};
} // namespace std
