//==------- builtins_scalar.hpp - SYCL scalar built-in functions -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lightweight alternative to <sycl/builtins.hpp> that provides only the
// scalar (float / double / half / integer fundamental-type) overloads of all
// SYCL built-in functions, without pulling in sycl::vec<> or sycl::marray<>
// template machinery.
//
// Trade-offs vs <sycl/builtins.hpp>:
//   - Significantly less front-end parse work (no vector.hpp, multi_ptr.hpp,
//     marray.hpp, vector_convert.hpp)
//   - Covers: sycl::sin(float), sycl::native::exp(float),
//             sycl::fmax(float,float), sycl::abs(int),
//             sycl::isequal(float,float), etc.
//   - Does NOT cover: sycl::sin(sycl::float4),
//             sycl::native::exp(sycl::marray<float,8>), etc.
//     Use <sycl/builtins.hpp> for those.
//
// Including both this header and <sycl/builtins.hpp> in the same translation
// unit is safe: the scalar inline functions are identical and ODR-compliant.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builtins/scalar_builtins.hpp>
