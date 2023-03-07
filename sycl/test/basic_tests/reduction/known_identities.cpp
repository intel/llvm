// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests the existance and specializations of known identities.

#include <sycl/sycl.hpp>

#include <cassert>
#include <limits>
#include <type_traits>

template <typename BinOp, typename OperandT> constexpr void checkNoIdentity() {
  static_assert(!sycl::has_known_identity<BinOp, OperandT>::value,
                "Operation should not have a known identity!");
  static_assert(!sycl::has_known_identity_v<BinOp, OperandT>,
                "Operation should not have a known identity!");
}

#define CHECK_IDENTITY(BINOP, OPERAND, EXPECTED)                               \
  static_assert(sycl::has_known_identity<BINOP, OPERAND>::value,               \
                "No trait specialization for known identity!");                \
  static_assert(sycl::has_known_identity_v<BINOP, OPERAND>,                    \
                "No trait specialization for known identity!");                \
  static_assert(sycl::known_identity<BINOP, OPERAND>::value == EXPECTED,       \
                "Identity does not match expected.");                          \
  static_assert(sycl::known_identity_v<BINOP, OPERAND> == EXPECTED,            \
                "Identity does not match expected.");

template <typename OperandT> constexpr void checkAll() {
  if constexpr (std::is_arithmetic_v<OperandT> ||
                std::is_same_v<std::remove_cv_t<OperandT>, sycl::half>) {
    CHECK_IDENTITY(sycl::plus<OperandT>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::plus<>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::multiplies<OperandT>, OperandT, OperandT{1});
    CHECK_IDENTITY(sycl::multiplies<>, OperandT, OperandT{1});
  } else {
    checkNoIdentity<sycl::plus<OperandT>, OperandT>();
    checkNoIdentity<sycl::plus<>, OperandT>();
    checkNoIdentity<sycl::multiplies<OperandT>, OperandT>();
    checkNoIdentity<sycl::multiplies<>, OperandT>();
  }

  if constexpr (std::is_integral_v<OperandT>) {
    CHECK_IDENTITY(sycl::bit_and<OperandT>, OperandT,
                   static_cast<OperandT>(~OperandT{}));
    CHECK_IDENTITY(sycl::bit_and<>, OperandT,
                   static_cast<OperandT>(~OperandT{}));
    CHECK_IDENTITY(sycl::bit_or<OperandT>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::bit_or<>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::bit_xor<OperandT>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::bit_xor<>, OperandT, OperandT{});
    CHECK_IDENTITY(sycl::minimum<OperandT>, OperandT,
                   std::numeric_limits<OperandT>::max());
    CHECK_IDENTITY(sycl::minimum<>, OperandT,
                   std::numeric_limits<OperandT>::max());
    CHECK_IDENTITY(sycl::maximum<OperandT>, OperandT,
                   std::numeric_limits<OperandT>::lowest());
    CHECK_IDENTITY(sycl::maximum<>, OperandT,
                   std::numeric_limits<OperandT>::lowest());
  } else {
    checkNoIdentity<sycl::bit_and<OperandT>, OperandT>();
    checkNoIdentity<sycl::bit_and<>, OperandT>();
    checkNoIdentity<sycl::bit_or<OperandT>, OperandT>();
    checkNoIdentity<sycl::bit_or<>, OperandT>();
    checkNoIdentity<sycl::bit_xor<OperandT>, OperandT>();
    checkNoIdentity<sycl::bit_xor<>, OperandT>();
  }

  // The implementation is relaxed about logical operators to allow implicit
  // conversions for logical operators, so negative checks are not used for this
  // case.
  if constexpr (std::is_same_v<std::remove_cv_t<OperandT>, bool>) {
    CHECK_IDENTITY(sycl::logical_and<OperandT>, OperandT, true);
    CHECK_IDENTITY(sycl::logical_and<>, OperandT, true);
    CHECK_IDENTITY(sycl::logical_or<OperandT>, OperandT, false);
    CHECK_IDENTITY(sycl::logical_or<>, OperandT, false);
  }

  if constexpr (std::is_floating_point_v<OperandT> ||
                std::is_same_v<std::remove_cv_t<OperandT>, sycl::half>) {
    CHECK_IDENTITY(sycl::minimum<OperandT>, OperandT,
                   std::numeric_limits<OperandT>::infinity());
    CHECK_IDENTITY(sycl::minimum<>, OperandT,
                   std::numeric_limits<OperandT>::infinity());
    CHECK_IDENTITY(sycl::maximum<OperandT>, OperandT,
                   -std::numeric_limits<OperandT>::infinity());
    CHECK_IDENTITY(sycl::maximum<>, OperandT,
                   -std::numeric_limits<OperandT>::infinity());
  }

  if constexpr (!std::is_integral_v<OperandT> &&
                !std::is_floating_point_v<OperandT> &&
                !std::is_same_v<std::remove_cv_t<OperandT>, sycl::half>) {
    checkNoIdentity<sycl::minimum<OperandT>, OperandT>();
    checkNoIdentity<sycl::minimum<>, OperandT>();
    checkNoIdentity<sycl::maximum<OperandT>, OperandT>();
    checkNoIdentity<sycl::maximum<>, OperandT>();
  }
}

struct CustomType {};

int main() {
  checkAll<bool>();
  checkAll<char>();
  checkAll<short>();
  checkAll<int>();
  checkAll<long>();
  checkAll<long long>();
  checkAll<signed char>();
  checkAll<unsigned char>();
  checkAll<unsigned int>();
  checkAll<unsigned long>();
  checkAll<unsigned long long>();
  checkAll<float>();
  checkAll<double>();
  checkAll<sycl::half>();
  checkAll<CustomType>();
  return 0;
}
