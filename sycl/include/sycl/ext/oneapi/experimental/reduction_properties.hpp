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

} // namespace detail

template <typename BufferT, typename BinaryOperation, typename PropertyList>
auto reduction(BufferT vars, handler &cgh, BinaryOperation combiner,
               PropertyList properties) {
  detail::CheckReductionIdentity<typename BufferT::value_type, BinaryOperation>(
      properties);
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(vars, cgh, WrappedOp, RuntimeProps);
}

template <typename T, typename BinaryOperation, typename PropertyList>
auto reduction(T *var, BinaryOperation combiner, PropertyList properties) {
  detail::CheckReductionIdentity<T, BinaryOperation>(properties);
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(var, WrappedOp, RuntimeProps);
}

template <typename T, size_t Extent, typename BinaryOperation,
          typename PropertyList>
auto reduction(span<T, Extent> vars, BinaryOperation combiner,
               PropertyList properties) {
  detail::CheckReductionIdentity<T, BinaryOperation>(properties);
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(vars, WrappedOp, RuntimeProps);
}

template <typename BufferT, typename BinaryOperation, typename PropertyList>
auto reduction(BufferT vars, handler &cgh,
               const typename BufferT::value_type &identity,
               BinaryOperation combiner, PropertyList properties) {
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(vars, cgh, identity, WrappedOp, RuntimeProps);
}

template <typename T, typename BinaryOperation, typename PropertyList>
auto reduction(T *var, const T &identity, BinaryOperation combiner,
               PropertyList properties) {
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(var, identity, WrappedOp, RuntimeProps);
}

template <typename T, size_t Extent, typename BinaryOperation,
          typename PropertyList>
auto reduction(span<T, Extent> vars, const T &identity,
               BinaryOperation combiner, PropertyList properties) {
  auto WrappedOp = detail::WrapOp(combiner, properties);
  auto RuntimeProps = detail::GetReductionPropertyList(properties);
  return reduction(vars, identity, WrappedOp, RuntimeProps);
}

} // namespace _V1
} // namespace sycl
