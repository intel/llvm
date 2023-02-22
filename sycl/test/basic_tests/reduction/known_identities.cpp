// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

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

template <typename BinOp, typename OperandT>
void checkIdentity(OperandT Expected) {
  static_assert(sycl::has_known_identity<BinOp, OperandT>::value,
                "No trait specialization for known identity!");
  static_assert(sycl::has_known_identity_v<BinOp, OperandT>,
                "No trait specialization for known identity!");
  assert((sycl::known_identity<BinOp, OperandT>::value == Expected) &&
         "Identity does not match expected.");
  assert((sycl::known_identity_v<BinOp, OperandT> == Expected) &&
         "Identity does not match expected.");
}

template <typename OperandT> constexpr void checkAll() {
  if constexpr (std::is_arithmetic_v<OperandT> ||
                std::is_same_v<std::remove_cv_t<OperandT>, sycl::half>) {
    checkIdentity<sycl::plus<OperandT>, OperandT>(OperandT{});
    checkIdentity<sycl::plus<>, OperandT>(OperandT{});
    checkIdentity<sycl::multiplies<OperandT>, OperandT>(OperandT{1});
    checkIdentity<sycl::multiplies<>, OperandT>(OperandT{1});
  } else {
    checkNoIdentity<sycl::plus<OperandT>, OperandT>();
    checkNoIdentity<sycl::plus<>, OperandT>();
    checkNoIdentity<sycl::multiplies<OperandT>, OperandT>();
    checkNoIdentity<sycl::multiplies<>, OperandT>();
  }

  if constexpr (std::is_integral_v<OperandT>) {
    checkIdentity<sycl::bit_and<OperandT>, OperandT>(~OperandT{});
    checkIdentity<sycl::bit_and<>, OperandT>(~OperandT{});
    checkIdentity<sycl::bit_or<OperandT>, OperandT>(OperandT{});
    checkIdentity<sycl::bit_or<>, OperandT>(OperandT{});
    checkIdentity<sycl::bit_xor<OperandT>, OperandT>(OperandT{});
    checkIdentity<sycl::bit_xor<>, OperandT>(OperandT{});
    checkIdentity<sycl::minimum<OperandT>, OperandT>(
        std::numeric_limits<OperandT>::max());
    checkIdentity<sycl::minimum<>, OperandT>(
        std::numeric_limits<OperandT>::max());
    checkIdentity<sycl::maximum<OperandT>, OperandT>(
        std::numeric_limits<OperandT>::lowest());
    checkIdentity<sycl::maximum<>, OperandT>(
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
    checkIdentity<sycl::logical_and<OperandT>, OperandT>(true);
    checkIdentity<sycl::logical_and<>, OperandT>(true);
    checkIdentity<sycl::logical_or<OperandT>, OperandT>(false);
    checkIdentity<sycl::logical_or<>, OperandT>(false);
  }

  if constexpr (std::is_floating_point_v<OperandT> ||
                std::is_same_v<std::remove_cv_t<OperandT>, sycl::half>) {
    checkIdentity<sycl::minimum<OperandT>, OperandT>(
        std::numeric_limits<OperandT>::infinity());
    checkIdentity<sycl::minimum<>, OperandT>(
        std::numeric_limits<OperandT>::infinity());
    checkIdentity<sycl::maximum<OperandT>, OperandT>(
        -std::numeric_limits<OperandT>::infinity());
    checkIdentity<sycl::maximum<>, OperandT>(
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
