//==----------- known_op_traits.hpp - Known binary operation traits --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header is intentionally lightweight: it must not include <functional>
// or any heavy SYCL header. It lets sycl/known_identity.hpp recognize binary
// operations without depending on the function object definitions; the
// function objects themselves register through specializations of
// known_op_traits in sycl/functional.hpp. As a consequence, an operation is
// only recognized as "known" (e.g. for has_known_identity) once
// sycl/functional.hpp has been included.

namespace sycl {
inline namespace _V1 {
namespace detail {

enum class known_op_kind {
  none,
  plus,
  multiplies,
  bit_and,
  bit_or,
  bit_xor,
  logical_and,
  logical_or,
  minimum,
  maximum
};

// Primary template: unknown operation.
template <typename BinaryOperation> struct known_op_traits {
  static constexpr known_op_kind kind = known_op_kind::none;
  using operand_type = void;
};

// Helper base for specializations. OperandT is the operand type of the
// function object, or void for transparent function objects.
template <known_op_kind Kind, typename OperandT> struct known_op_traits_base {
  static constexpr known_op_kind kind = Kind;
  using operand_type = OperandT;
};

// Forward declaration for deterministic reductions.
template <typename BinaryOperation> struct DeterministicOperatorWrapper;

// A deterministic wrapper around a known operation is the same known
// operation. Only a single level of wrapping is recognized.
template <typename BinaryOperation>
struct known_op_traits<DeterministicOperatorWrapper<BinaryOperation>>
    : known_op_traits<BinaryOperation> {};
template <typename BinaryOperation>
struct known_op_traits<
    DeterministicOperatorWrapper<DeterministicOperatorWrapper<BinaryOperation>>>
    : known_op_traits<void> {};

} // namespace detail
} // namespace _V1
} // namespace sycl
