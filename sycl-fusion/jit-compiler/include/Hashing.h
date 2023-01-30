//==---- Hashing.h - helper for hashes for JIT internal representations ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_HASHING_H
#define SYCL_FUSION_JIT_COMPILER_HASHING_H

#include "Parameter.h"

#include "llvm/ADT/Hashing.h"

#include <tuple>
#include <vector>

namespace jit_compiler {
inline llvm::hash_code hash_value(const ParameterInternalization &P) {
  return llvm::hash_combine(P.LocalSize, P.Intern, P.Param);
}

inline llvm::hash_code hash_value(const Parameter &P) {
  return llvm::hash_combine(P.ParamIdx, P.KernelIdx);
}

inline llvm::hash_code hash_value(const JITConstant &C) {
  return llvm::hash_combine(C.Param, C.Value);
}

inline llvm::hash_code hash_value(const ParameterIdentity &IP) {
  return llvm::hash_combine(IP.LHS, IP.RHS);
}
} // namespace jit_compiler

namespace std {
template <typename T> inline llvm::hash_code hash_value(const vector<T> &V) {
  return llvm::hash_combine_range(V.begin(), V.end());
}

template <typename... T> struct hash<tuple<T...>> {
  size_t operator()(const tuple<T...> &Tuple) const noexcept {
    return llvm::hash_value(Tuple);
  }
};
} // namespace std

#endif // SYCL_FUSION_JIT_COMPILER_HASHING_H
