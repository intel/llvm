//==------- properties.hpp - SYCL properties associated with reductions ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#define SYCL_EXT_ONEAPI_REDUCTION_PROPERTIES

#include <sycl/ext/oneapi/properties/property.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>
#include <sycl/reduction.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

struct deterministic_key
    : detail::compile_time_property_key<detail::PropKind::Deterministic> {
  using value_t = property_value<deterministic_key>;
};
inline constexpr deterministic_key::value_t deterministic;

struct initialize_to_identity_key
    : detail::compile_time_property_key<
          detail::PropKind::InitializeToIdentity> {
  using value_t = property_value<initialize_to_identity_key>;
};
inline constexpr initialize_to_identity_key::value_t initialize_to_identity;

namespace detail {
struct reduction_property_check_anchor {};
} // namespace detail

template <>
struct is_property_key_of<deterministic_key,
                          detail::reduction_property_check_anchor>
    : std::true_type {};

template <>
struct is_property_key_of<initialize_to_identity_key,
                          detail::reduction_property_check_anchor>
    : std::true_type {};

} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {

template <typename BinaryOperation, typename PropertyList>
auto WrapOp(BinaryOperation combiner, PropertyList properties) {
  if constexpr (properties.template has_property<
                    ext::oneapi::experimental::deterministic_key>()) {
    return DeterministicOperatorWrapper(combiner);
  } else {
    return combiner;
  }
}

template <typename T, typename BinaryOperation, typename PropertyList>
void CheckReductionIdentity(PropertyList properties) {
  if constexpr (properties.template has_property<
                    ext::oneapi::experimental::initialize_to_identity_key>()) {
    static_assert(has_known_identity_v<BinaryOperation, T>,
                  "initialize_to_identity requires an identity value.");
  }
}

template <typename PropertyList>
property_list GetReductionPropertyList(PropertyList properties) {
  if constexpr (properties.template has_property<
                    ext::oneapi::experimental::initialize_to_identity_key>()) {
    return sycl::property::reduction::initialize_to_identity{};
  }
  return {};
}

template <typename BinaryOperation> struct DeterministicOperatorWrapper {

  DeterministicOperatorWrapper(BinaryOperation BinOp = BinaryOperation())
      : BinOp(BinOp) {}

  template <typename... Args>
  std::invoke_result_t<BinaryOperation, Args...> operator()(Args... args) {
    return BinOp(std::forward<Args>(args)...);
  }

  BinaryOperation BinOp;
};

template <typename BinaryOperation>
struct IsDeterministicOperator<DeterministicOperatorWrapper<BinaryOperation>>
    : std::true_type {};

template <typename PropertyList>
inline constexpr bool is_valid_reduction_prop_list =
    ext::oneapi::experimental::detail::all_are_properties_of_v<
        ext::oneapi::experimental::detail::reduction_property_check_anchor,
        PropertyList>;

template <typename BinaryOperation, typename PropertyList, typename... Args>
auto convert_reduction_properties(BinaryOperation combiner,
                                  PropertyList properties, Args &&...args) {
  if constexpr (is_valid_reduction_prop_list<PropertyList>) {
    auto WrappedOp = WrapOp(combiner, properties);
    auto RuntimeProps = GetReductionPropertyList(properties);
    return sycl::reduction(std::forward<Args>(args)..., WrappedOp,
                           RuntimeProps);
  } else {
    // Invalid, will be disabled by SFINAE at the caller side. Make sure no hard
    // error is emitted from here.
  }
}
} // namespace detail

template <typename BufferT, typename BinaryOperation, typename PropertyList>
auto reduction(BufferT vars, handler &cgh, BinaryOperation combiner,
               PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, vars, cgh))> {
  detail::CheckReductionIdentity<typename BufferT::value_type, BinaryOperation>(
      properties);
  return detail::convert_reduction_properties(combiner, properties, vars, cgh);
}

template <typename T, typename BinaryOperation, typename PropertyList>
auto reduction(T *var, BinaryOperation combiner, PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, var))> {
  detail::CheckReductionIdentity<T, BinaryOperation>(properties);
  return detail::convert_reduction_properties(combiner, properties, var);
}

template <typename T, size_t Extent, typename BinaryOperation,
          typename PropertyList>
auto reduction(span<T, Extent> vars, BinaryOperation combiner,
               PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, vars))> {
  detail::CheckReductionIdentity<T, BinaryOperation>(properties);
  return detail::convert_reduction_properties(combiner, properties, vars);
}

template <typename BufferT, typename BinaryOperation, typename PropertyList>
auto reduction(BufferT vars, handler &cgh,
               const typename BufferT::value_type &identity,
               BinaryOperation combiner, PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, vars, cgh, identity))> {
  return detail::convert_reduction_properties(combiner, properties, vars, cgh,
                                              identity);
}

template <typename T, typename BinaryOperation, typename PropertyList>
auto reduction(T *var, const T &identity, BinaryOperation combiner,
               PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, var, identity))> {
  return detail::convert_reduction_properties(combiner, properties, var,
                                              identity);
}

template <typename T, size_t Extent, typename BinaryOperation,
          typename PropertyList>
auto reduction(span<T, Extent> vars, const T &identity,
               BinaryOperation combiner, PropertyList properties)
    -> std::enable_if_t<detail::is_valid_reduction_prop_list<PropertyList>,
                        decltype(detail::convert_reduction_properties(
                            combiner, properties, vars, identity))> {
  return detail::convert_reduction_properties(combiner, properties, vars,
                                              identity);
}

} // namespace _V1
} // namespace sycl
