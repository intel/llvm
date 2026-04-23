//==----------- kernel_properties.hpp - SYCL kernel properties ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <stddef.h>
#include <stdint.h>
#include <sycl/aspects.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/ext/oneapi/properties.hpp>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct properties_tag {};

struct device_has_key
		: detail::compile_time_property_key<detail::PropKind::DeviceHas> {
	template <aspect... Aspects>
	using value_t = property_value<device_has_key,
																 std::integral_constant<aspect, Aspects>...>;
};

template <aspect... Aspects>
struct property_value<device_has_key,
											std::integral_constant<aspect, Aspects>...>
		: detail::property_base<
					property_value<device_has_key,
												 std::integral_constant<aspect, Aspects>...>,
					detail::PropKind::DeviceHas, device_has_key> {
	static constexpr std::array<aspect, sizeof...(Aspects)> value{Aspects...};
};

template <aspect... Aspects>
inline constexpr device_has_key::value_t<Aspects...> device_has;

struct work_group_progress_key
		: detail::compile_time_property_key<detail::PropKind::WorkGroupProgress> {
	template <forward_progress_guarantee Guarantee,
						execution_scope CoordinationScope>
	using value_t = property_value<
			work_group_progress_key,
			std::integral_constant<forward_progress_guarantee, Guarantee>,
			std::integral_constant<execution_scope, CoordinationScope>>;
};

struct sub_group_progress_key
		: detail::compile_time_property_key<detail::PropKind::SubGroupProgress> {
	template <forward_progress_guarantee Guarantee,
						execution_scope CoordinationScope>
	using value_t = property_value<
			sub_group_progress_key,
			std::integral_constant<forward_progress_guarantee, Guarantee>,
			std::integral_constant<execution_scope, CoordinationScope>>;
};

struct work_item_progress_key
		: detail::compile_time_property_key<detail::PropKind::WorkItemProgress> {
	template <forward_progress_guarantee Guarantee,
						execution_scope CoordinationScope>
	using value_t = property_value<
			work_item_progress_key,
			std::integral_constant<forward_progress_guarantee, Guarantee>,
			std::integral_constant<execution_scope, CoordinationScope>>;
};

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
struct property_value<
		work_group_progress_key,
		std::integral_constant<forward_progress_guarantee, Guarantee>,
		std::integral_constant<execution_scope, CoordinationScope>>
		: detail::property_base<
					property_value<
							work_group_progress_key,
							std::integral_constant<forward_progress_guarantee, Guarantee>,
							std::integral_constant<execution_scope, CoordinationScope>>,
					detail::PropKind::WorkGroupProgress, work_group_progress_key> {
	static constexpr forward_progress_guarantee guarantee = Guarantee;
	static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
struct property_value<
		sub_group_progress_key,
		std::integral_constant<forward_progress_guarantee, Guarantee>,
		std::integral_constant<execution_scope, CoordinationScope>>
		: detail::property_base<
					property_value<
							sub_group_progress_key,
							std::integral_constant<forward_progress_guarantee, Guarantee>,
							std::integral_constant<execution_scope, CoordinationScope>>,
					detail::PropKind::SubGroupProgress, sub_group_progress_key> {
	static constexpr forward_progress_guarantee guarantee = Guarantee;
	static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
struct property_value<
		work_item_progress_key,
		std::integral_constant<forward_progress_guarantee, Guarantee>,
		std::integral_constant<execution_scope, CoordinationScope>>
		: detail::property_base<
					property_value<
							work_item_progress_key,
							std::integral_constant<forward_progress_guarantee, Guarantee>,
							std::integral_constant<execution_scope, CoordinationScope>>,
					detail::PropKind::WorkItemProgress, work_item_progress_key> {
	static constexpr forward_progress_guarantee guarantee = Guarantee;
	static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
inline constexpr work_group_progress_key::value_t<Guarantee, CoordinationScope>
		work_group_progress;

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
inline constexpr sub_group_progress_key::value_t<Guarantee, CoordinationScope>
		sub_group_progress;

template <forward_progress_guarantee Guarantee,
					execution_scope CoordinationScope>
inline constexpr work_item_progress_key::value_t<Guarantee, CoordinationScope>
		work_item_progress;

namespace detail {
template <sycl::aspect... Aspects>
struct HasCompileTimeEffect<device_has_key::value_t<Aspects...>>
		: std::true_type {};
template <aspect... Aspects>
struct PropertyMetaInfo<device_has_key::value_t<Aspects...>> {
	static constexpr const char *name = "sycl-device-has";
	static constexpr const char *value =
			SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};
template <aspect... Aspects>
struct FunctionPropertyMetaInfo<device_has_key::value_t<Aspects...>> {
	static constexpr const char *name = "sycl-device-has";
	static constexpr const char *value =
			SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};

template <typename T, typename = void>
struct HasKernelPropertiesGetMethod : std::false_type {};

template <typename T>
struct HasKernelPropertiesGetMethod<T,
																		std::void_t<decltype(std::declval<T>().get(
																				std::declval<properties_tag>()))>>
		: std::true_type {
	using properties_t =
			decltype(std::declval<T>().get(std::declval<properties_tag>()));
};

template <typename... RestT>
auto RetrieveGetMethodPropertiesOrEmpty(RestT &&...Rest) {
	auto Identity = [](const auto &x) -> decltype(auto) { return x; };
	const auto &KernelObj = (Identity(Rest), ...);
	if constexpr (ext::oneapi::experimental::detail::HasKernelPropertiesGetMethod<
										decltype(KernelObj)>::value) {
		return KernelObj.get(ext::oneapi::experimental::properties_tag{});
	} else {
		return ext::oneapi::experimental::empty_properties_t{};
	}
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl