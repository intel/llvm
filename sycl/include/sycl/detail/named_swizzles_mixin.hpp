//==---------------- named_swizzles_mixin.hpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This files implements two mixins
// `NamedSwizzlesMixinConst`/`NamedSwizzlesMixinBoth` that abstract away named
// swizzles implementation for SYCL vector and swizzles classes

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Will be defined in another header.
template <typename T> struct from_incomplete;

#ifndef SYCL_SIMPLE_SWIZZLES
#define __SYCL_SWIZZLE_MIXIN_SIMPLE_SWIZZLES
#else
// TODO: It might be beneficial to use partial specializations for different Ns,
// instead of making all the named swizzles templates with SFINAE conditions.
#define __SYCL_SWIZZLE_MIXIN_SIMPLE_SWIZZLES                                   \
  /* __swizzled_vec__ XYZW_SWIZZLE() const; */                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N <= 4, xx, 0, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xy, 0, 1)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xz, 0, 2)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xw, 0, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yx, 1, 0)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yy, 1, 1)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yz, 1, 2)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yw, 1, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zx, 2, 0)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zy, 2, 1)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zz, 2, 2)                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zw, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wx, 3, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wy, 3, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wz, 3, 2)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ww, 3, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N <= 4, xxx, 0, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xxy, 0, 0, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxz, 0, 0, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxw, 0, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyx, 0, 1, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyy, 0, 1, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyz, 0, 1, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xyw, 0, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzx, 0, 2, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzy, 0, 2, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzz, 0, 2, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzw, 0, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwx, 0, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwy, 0, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwz, 0, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xww, 0, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxx, 1, 0, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxy, 1, 0, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxz, 1, 0, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxw, 1, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyx, 1, 1, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyy, 1, 1, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyz, 1, 1, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yyw, 1, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzx, 1, 2, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzy, 1, 2, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzz, 1, 2, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzw, 1, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywx, 1, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywy, 1, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywz, 1, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yww, 1, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxx, 2, 0, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxy, 2, 0, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxz, 2, 0, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxw, 2, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyx, 2, 1, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyy, 2, 1, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyz, 2, 1, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zyw, 2, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzx, 2, 2, 0)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzy, 2, 2, 1)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzz, 2, 2, 2)                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzw, 2, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwx, 2, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwy, 2, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwz, 2, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zww, 2, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxx, 3, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxy, 3, 0, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxz, 3, 0, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxw, 3, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyx, 3, 1, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyy, 3, 1, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyz, 3, 1, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyw, 3, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzx, 3, 2, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzy, 3, 2, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzz, 3, 2, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzw, 3, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwx, 3, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwy, 3, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwz, 3, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, www, 3, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N <= 4, xxxx, 0, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xxxy, 0, 0, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxxz, 0, 0, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxxw, 0, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xxyx, 0, 0, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xxyy, 0, 0, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxyz, 0, 0, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxyw, 0, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxzx, 0, 0, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxzy, 0, 0, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xxzz, 0, 0, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxzw, 0, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxwx, 0, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxwy, 0, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxwz, 0, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xxww, 0, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyxx, 0, 1, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyxy, 0, 1, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyxz, 0, 1, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xyxw, 0, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyyx, 0, 1, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, xyyy, 0, 1, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyyz, 0, 1, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xyyw, 0, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyzx, 0, 1, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyzy, 0, 1, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xyzz, 0, 1, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xyzw, 0, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xywx, 0, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xywy, 0, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xywz, 0, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xyww, 0, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzxx, 0, 2, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzxy, 0, 2, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzxz, 0, 2, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzxw, 0, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzyx, 0, 2, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzyy, 0, 2, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzyz, 0, 2, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzyw, 0, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzzx, 0, 2, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzzy, 0, 2, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, xzzz, 0, 2, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzzw, 0, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzwx, 0, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzwy, 0, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzwz, 0, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xzww, 0, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwxx, 0, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwxy, 0, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwxz, 0, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwxw, 0, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwyx, 0, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwyy, 0, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwyz, 0, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwyw, 0, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwzx, 0, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwzy, 0, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwzz, 0, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwzw, 0, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwwx, 0, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwwy, 0, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwwz, 0, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, xwww, 0, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxxx, 1, 0, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxxy, 1, 0, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxxz, 1, 0, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxxw, 1, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxyx, 1, 0, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yxyy, 1, 0, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxyz, 1, 0, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxyw, 1, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxzx, 1, 0, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxzy, 1, 0, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yxzz, 1, 0, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxzw, 1, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxwx, 1, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxwy, 1, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxwz, 1, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yxww, 1, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyxx, 1, 1, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyxy, 1, 1, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyxz, 1, 1, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yyxw, 1, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyyx, 1, 1, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(2 <= N && N <= 4, yyyy, 1, 1, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyyz, 1, 1, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yyyw, 1, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyzx, 1, 1, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyzy, 1, 1, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yyzz, 1, 1, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yyzw, 1, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yywx, 1, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yywy, 1, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yywz, 1, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yyww, 1, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzxx, 1, 2, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzxy, 1, 2, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzxz, 1, 2, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzxw, 1, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzyx, 1, 2, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzyy, 1, 2, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzyz, 1, 2, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzyw, 1, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzzx, 1, 2, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzzy, 1, 2, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, yzzz, 1, 2, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzzw, 1, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzwx, 1, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzwy, 1, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzwz, 1, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, yzww, 1, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywxx, 1, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywxy, 1, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywxz, 1, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywxw, 1, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywyx, 1, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywyy, 1, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywyz, 1, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywyw, 1, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywzx, 1, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywzy, 1, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywzz, 1, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywzw, 1, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywwx, 1, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywwy, 1, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywwz, 1, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ywww, 1, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxxx, 2, 0, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxxy, 2, 0, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxxz, 2, 0, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxxw, 2, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxyx, 2, 0, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxyy, 2, 0, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxyz, 2, 0, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxyw, 2, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxzx, 2, 0, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxzy, 2, 0, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zxzz, 2, 0, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxzw, 2, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxwx, 2, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxwy, 2, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxwz, 2, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zxww, 2, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyxx, 2, 1, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyxy, 2, 1, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyxz, 2, 1, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zyxw, 2, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyyx, 2, 1, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyyy, 2, 1, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyyz, 2, 1, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zyyw, 2, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyzx, 2, 1, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyzy, 2, 1, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zyzz, 2, 1, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zyzw, 2, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zywx, 2, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zywy, 2, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zywz, 2, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zyww, 2, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzxx, 2, 2, 0, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzxy, 2, 2, 0, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzxz, 2, 2, 0, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzxw, 2, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzyx, 2, 2, 1, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzyy, 2, 2, 1, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzyz, 2, 2, 1, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzyw, 2, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzzx, 2, 2, 2, 0)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzzy, 2, 2, 2, 1)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, zzzz, 2, 2, 2, 2)              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzzw, 2, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzwx, 2, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzwy, 2, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzwz, 2, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zzww, 2, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwxx, 2, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwxy, 2, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwxz, 2, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwxw, 2, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwyx, 2, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwyy, 2, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwyz, 2, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwyw, 2, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwzx, 2, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwzy, 2, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwzz, 2, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwzw, 2, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwwx, 2, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwwy, 2, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwwz, 2, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, zwww, 2, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxxx, 3, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxxy, 3, 0, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxxz, 3, 0, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxxw, 3, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxyx, 3, 0, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxyy, 3, 0, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxyz, 3, 0, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxyw, 3, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxzx, 3, 0, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxzy, 3, 0, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxzz, 3, 0, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxzw, 3, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxwx, 3, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxwy, 3, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxwz, 3, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wxww, 3, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyxx, 3, 1, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyxy, 3, 1, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyxz, 3, 1, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyxw, 3, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyyx, 3, 1, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyyy, 3, 1, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyyz, 3, 1, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyyw, 3, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyzx, 3, 1, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyzy, 3, 1, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyzz, 3, 1, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyzw, 3, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wywx, 3, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wywy, 3, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wywz, 3, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wyww, 3, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzxx, 3, 2, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzxy, 3, 2, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzxz, 3, 2, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzxw, 3, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzyx, 3, 2, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzyy, 3, 2, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzyz, 3, 2, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzyw, 3, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzzx, 3, 2, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzzy, 3, 2, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzzz, 3, 2, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzzw, 3, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzwx, 3, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzwy, 3, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzwz, 3, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wzww, 3, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwxx, 3, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwxy, 3, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwxz, 3, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwxw, 3, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwyx, 3, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwyy, 3, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwyz, 3, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwyw, 3, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwzx, 3, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwzy, 3, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwzz, 3, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwzw, 3, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwwx, 3, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwwy, 3, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwwz, 3, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, wwww, 3, 3, 3, 3)                        \
                                                                               \
  /* __swizzled_vec__ RGBA_SWIZZLE() const; */                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rr, 0, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rg, 0, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rb, 0, 2)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ra, 0, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gr, 1, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gg, 1, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gb, 1, 2)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ga, 1, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, br, 2, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bg, 2, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bb, 2, 2)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ba, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ar, 3, 0)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ag, 3, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ab, 3, 2)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aa, 3, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrr, 0, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrg, 0, 0, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrb, 0, 0, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rra, 0, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgr, 0, 1, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgg, 0, 1, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgb, 0, 1, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rga, 0, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbr, 0, 2, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbg, 0, 2, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbb, 0, 2, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rba, 0, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rar, 0, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rag, 0, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rab, 0, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raa, 0, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grr, 1, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grg, 1, 0, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grb, 1, 0, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gra, 1, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggr, 1, 1, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggg, 1, 1, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggb, 1, 1, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gga, 1, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbr, 1, 2, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbg, 1, 2, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbb, 1, 2, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gba, 1, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gar, 1, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gag, 1, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gab, 1, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaa, 1, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brr, 2, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brg, 2, 0, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brb, 2, 0, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bra, 2, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgr, 2, 1, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgg, 2, 1, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgb, 2, 1, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bga, 2, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbr, 2, 2, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbg, 2, 2, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbb, 2, 2, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bba, 2, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bar, 2, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bag, 2, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bab, 2, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baa, 2, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arr, 3, 0, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arg, 3, 0, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arb, 3, 0, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ara, 3, 0, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agr, 3, 1, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agg, 3, 1, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agb, 3, 1, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aga, 3, 1, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abr, 3, 2, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abg, 3, 2, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abb, 3, 2, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aba, 3, 2, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aar, 3, 3, 0)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aag, 3, 3, 1)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aab, 3, 3, 2)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaa, 3, 3, 3)                            \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrrr, 0, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrrg, 0, 0, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrrb, 0, 0, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrra, 0, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrgr, 0, 0, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrgg, 0, 0, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrgb, 0, 0, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrga, 0, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrbr, 0, 0, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrbg, 0, 0, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrbb, 0, 0, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrba, 0, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrar, 0, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrag, 0, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rrab, 0, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rraa, 0, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgrr, 0, 1, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgrg, 0, 1, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgrb, 0, 1, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgra, 0, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rggr, 0, 1, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rggg, 0, 1, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rggb, 0, 1, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgga, 0, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgbr, 0, 1, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgbg, 0, 1, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgbb, 0, 1, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgba, 0, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgar, 0, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgag, 0, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgab, 0, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rgaa, 0, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbrr, 0, 2, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbrg, 0, 2, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbrb, 0, 2, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbra, 0, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbgr, 0, 2, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbgg, 0, 2, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbgb, 0, 2, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbga, 0, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbbr, 0, 2, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbbg, 0, 2, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbbb, 0, 2, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbba, 0, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbar, 0, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbag, 0, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbab, 0, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rbaa, 0, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rarr, 0, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rarg, 0, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rarb, 0, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rara, 0, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ragr, 0, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ragg, 0, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ragb, 0, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raga, 0, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rabr, 0, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rabg, 0, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, rabb, 0, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raba, 0, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raar, 0, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raag, 0, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raab, 0, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, raaa, 0, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grrr, 1, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grrg, 1, 0, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grrb, 1, 0, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grra, 1, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grgr, 1, 0, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grgg, 1, 0, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grgb, 1, 0, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grga, 1, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grbr, 1, 0, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grbg, 1, 0, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grbb, 1, 0, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grba, 1, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grar, 1, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grag, 1, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, grab, 1, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, graa, 1, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggrr, 1, 1, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggrg, 1, 1, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggrb, 1, 1, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggra, 1, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gggr, 1, 1, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gggg, 1, 1, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gggb, 1, 1, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggga, 1, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggbr, 1, 1, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggbg, 1, 1, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggbb, 1, 1, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggba, 1, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggar, 1, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggag, 1, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggab, 1, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, ggaa, 1, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbrr, 1, 2, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbrg, 1, 2, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbrb, 1, 2, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbra, 1, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbgr, 1, 2, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbgg, 1, 2, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbgb, 1, 2, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbga, 1, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbbr, 1, 2, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbbg, 1, 2, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbbb, 1, 2, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbba, 1, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbar, 1, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbag, 1, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbab, 1, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gbaa, 1, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, garr, 1, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, garg, 1, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, garb, 1, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gara, 1, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gagr, 1, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gagg, 1, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gagb, 1, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaga, 1, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gabr, 1, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gabg, 1, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gabb, 1, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaba, 1, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaar, 1, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaag, 1, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaab, 1, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, gaaa, 1, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brrr, 2, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brrg, 2, 0, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brrb, 2, 0, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brra, 2, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brgr, 2, 0, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brgg, 2, 0, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brgb, 2, 0, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brga, 2, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brbr, 2, 0, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brbg, 2, 0, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brbb, 2, 0, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brba, 2, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brar, 2, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brag, 2, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, brab, 2, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, braa, 2, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgrr, 2, 1, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgrg, 2, 1, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgrb, 2, 1, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgra, 2, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bggr, 2, 1, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bggg, 2, 1, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bggb, 2, 1, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgga, 2, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgbr, 2, 1, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgbg, 2, 1, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgbb, 2, 1, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgba, 2, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgar, 2, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgag, 2, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgab, 2, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bgaa, 2, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbrr, 2, 2, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbrg, 2, 2, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbrb, 2, 2, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbra, 2, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbgr, 2, 2, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbgg, 2, 2, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbgb, 2, 2, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbga, 2, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbbr, 2, 2, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbbg, 2, 2, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbbb, 2, 2, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbba, 2, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbar, 2, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbag, 2, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbab, 2, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bbaa, 2, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, barr, 2, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, barg, 2, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, barb, 2, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bara, 2, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bagr, 2, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bagg, 2, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, bagb, 2, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baga, 2, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, babr, 2, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, babg, 2, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, babb, 2, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baba, 2, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baar, 2, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baag, 2, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baab, 2, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, baaa, 2, 3, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arrr, 3, 0, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arrg, 3, 0, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arrb, 3, 0, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arra, 3, 0, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, argr, 3, 0, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, argg, 3, 0, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, argb, 3, 0, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arga, 3, 0, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arbr, 3, 0, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arbg, 3, 0, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arbb, 3, 0, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arba, 3, 0, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arar, 3, 0, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arag, 3, 0, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, arab, 3, 0, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, araa, 3, 0, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agrr, 3, 1, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agrg, 3, 1, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agrb, 3, 1, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agra, 3, 1, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aggr, 3, 1, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aggg, 3, 1, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aggb, 3, 1, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agga, 3, 1, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agbr, 3, 1, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agbg, 3, 1, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agbb, 3, 1, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agba, 3, 1, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agar, 3, 1, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agag, 3, 1, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agab, 3, 1, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, agaa, 3, 1, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abrr, 3, 2, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abrg, 3, 2, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abrb, 3, 2, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abra, 3, 2, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abgr, 3, 2, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abgg, 3, 2, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abgb, 3, 2, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abga, 3, 2, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abbr, 3, 2, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abbg, 3, 2, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abbb, 3, 2, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abba, 3, 2, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abar, 3, 2, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abag, 3, 2, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abab, 3, 2, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, abaa, 3, 2, 3, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aarr, 3, 3, 0, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aarg, 3, 3, 0, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aarb, 3, 3, 0, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aara, 3, 3, 0, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aagr, 3, 3, 1, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aagg, 3, 3, 1, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aagb, 3, 3, 1, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaga, 3, 3, 1, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aabr, 3, 3, 2, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aabg, 3, 3, 2, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aabb, 3, 3, 2, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaba, 3, 3, 2, 3)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaar, 3, 3, 3, 0)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaag, 3, 3, 3, 1)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaab, 3, 3, 3, 2)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, aaaa, 3, 3, 3, 3)
#endif

#define __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES                                      \
  /* __swizzled_vec__ XYZW_ACCESS() const; */                                  \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N <= 4, x, 0)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 2 || N == 3 || N == 4, y, 1)         \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 3 || N == 4, z, 2)                   \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 4, w, 3)                             \
                                                                               \
  /* __swizzled_vec__ RGBA_ACCESS() const; */                                  \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 4, r, 0)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 4, g, 1)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 4, b, 2)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 4, a, 3)                             \
                                                                               \
  /* __swizzled_vec__ INDEX_ACCESS() const; */                                 \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 0, s0, 0)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 1, s1, 1)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 2, s2, 2)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 2, s3, 3)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 4, s4, 4)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 4, s5, 5)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 4, s6, 6)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N > 4, s7, 7)                             \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, s8, 8)                           \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, s9, 9)                           \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sA, 10)                          \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sB, 11)                          \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sC, 12)                          \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sD, 13)                          \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sE, 14)                          \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(N == 16, sF, 15)                          \
                                                                               \
  /* __swizzled_vec__ lo()/hi() const; */                                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, lo, 0)                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, lo, 0, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, lo, 0, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, lo, 0, 1, 2, 3)                          \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, lo, 0, 1, 2, 3, 4, 5, 6, 7)             \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, hi, 1)                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, hi, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, hi, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, hi, 4, 5, 6, 7)                          \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, hi, 8, 9, 10, 11, 12, 13, 14, 15)       \
  /* __swizzled_vec__ odd()/even() const; */                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, odd, 1)                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, odd, 1, 3)                               \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, odd, 1, 3)                               \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, odd, 1, 3, 5, 7)                         \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, odd, 1, 3, 5, 7, 9, 11, 13, 15)         \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, even, 0)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, even, 0, 2)                              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, even, 0, 2)                              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, even, 0, 2, 4, 6)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, even, 0, 2, 4, 6, 8, 10, 12, 14)        \
  /* SYCL_SIMPLE_SWIZZLES */                                                   \
  __SYCL_SWIZZLE_MIXIN_SIMPLE_SWIZZLES

#define __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST(COND, NAME, ...)                 \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<                                                            \
      (COND), decltype(std::declval<Self_>().template swizzle<__VA_ARGS__>())> \
  NAME() {                                                                     \
    return static_cast<Self_ *>(this)->template swizzle<__VA_ARGS__>();        \
  }

#define __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, ...)                     \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<(COND), decltype(std::declval<const Self_>()                \
                                        .template swizzle<__VA_ARGS__>())>     \
  NAME() const {                                                               \
    return static_cast<const Self_ *>(this)->template swizzle<__VA_ARGS__>();  \
  }

#define __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS_NON_CONST(COND, NAME, INDEX)        \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<(COND), decltype(std::declval<Self_>()[0])> NAME() {        \
    return (*static_cast<Self_ *>(this))[INDEX];                               \
  }
#define __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS_CONST(COND, NAME, INDEX)            \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<(COND), decltype(std::declval<const Self_>()[0])> NAME()    \
      const {                                                                  \
    return (*static_cast<const Self_ *>(this))[INDEX];                         \
  }

template <typename Self, int NumElements = from_incomplete<Self>::size()>
struct NamedSwizzlesMixinConst {
#define __SYCL_SWIZZLE_MIXIN_METHOD(COND, NAME, ...)                           \
  __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, __VA_ARGS__)

#define __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(COND, NAME, INDEX)                  \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS_CONST(COND, NAME, INDEX)

  __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES

#undef __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS
#undef __SYCL_SWIZZLE_MIXIN_METHOD
};

template <typename Self, int NumElements = from_incomplete<Self>::size()>
struct NamedSwizzlesMixinBoth {
#define __SYCL_SWIZZLE_MIXIN_METHOD(COND, NAME, ...)                           \
  __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST(COND, NAME, __VA_ARGS__)               \
  __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, __VA_ARGS__)

#define __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS(COND, NAME, INDEX)                  \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS_NON_CONST(COND, NAME, INDEX)              \
  __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS_CONST(COND, NAME, INDEX)

  __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES

#undef __SYCL_SWIZLLE_MIXIN_SCALAR_ACCESS
#undef __SYCL_SWIZZLE_MIXIN_METHOD
};

#undef __SYCL_SWIZZLE_MIXIN_METHOD_CONST
#undef __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST

#undef __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES
#undef __SYCL_SWIZZLE_MIXIN_SIMPLE_SWIZZLES

} // namespace detail
} // namespace _V1
} // namespace sycl
