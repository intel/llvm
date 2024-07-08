/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility extension
 *
 *  launch.hpp
 *
 *  Description:
 *    launch functionality for the SYCL compatibility extension
 **************************************************************************/

#pragma once

#include "sycl/ext/oneapi/properties/properties.hpp"
#include "sycl/ext/oneapi/experimental/enqueue_functions.hpp"
#include <sycl/event.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/traits.hpp>
#include <syclcompat/defs.hpp>

namespace syclcompat {
namespace experimental {

namespace sycl_exp = sycl::ext::oneapi::experimental;

template <typename Properties> struct kernel_properties {
  template <typename... Props>
  kernel_properties(Props... properties) : props{properties...} {}
  using Props = Properties;
  Properties props;
};

template <typename... Props>
kernel_properties(Props... props) -> kernel_properties<decltype(sycl_exp::properties(props...))>;

template <typename Properties> struct launch_properties {
  template <typename... Props>
  launch_properties(Props... properties) : props{properties...} {}
  using Props = Properties;
  Properties props;
};

template <typename... Props>
launch_properties(Props... props) -> launch_properties<decltype(sycl_exp::properties(props...))>;

struct local_mem_size {
  local_mem_size(size_t size) : size{size} {};
  local_mem_size() : size{} {};
  size_t size;
};

namespace detail{
template <typename T> struct is_kernel_properties : std::false_type{};
template <typename TT> struct is_kernel_properties<kernel_properties<TT>> : std::true_type{};

template <typename T> struct is_launch_properties : std::false_type{};
template <typename TT> struct is_launch_properties<launch_properties<TT>> : std::true_type{};

template <typename T> struct is_local_mem_size : std::false_type{};
template <> struct is_local_mem_size<local_mem_size> : std::true_type{};

template <typename T, typename Tuple> struct tuple_element_index_helper;

template<typename T>
struct tuple_element_index_helper<T, std::tuple<>>
{
  static constexpr std::size_t value = 0;
};

template<typename T, typename... Rest>
struct tuple_element_index_helper<T, std::tuple<T, Rest...>>
{
  static constexpr std::size_t value = 0;
  using RestTuple = std::tuple<Rest...>;
  static_assert(
    tuple_element_index_helper<T, RestTuple>::value == 
    std::tuple_size_v<RestTuple>,
    "type appears more than once in tuple");
};

template<typename T, typename First, typename... Rest>
struct tuple_element_index_helper<T, std::tuple<First, Rest...>>
{
  using RestTuple = std::tuple<Rest...>;
  static constexpr std::size_t value = 1 +
       tuple_element_index_helper<T, RestTuple>::value;
};

template<typename T, typename Tuple>
struct tuple_element_index
{
  static constexpr std::size_t value =
    tuple_element_index_helper<T, Tuple>::value;
  static_assert(value < std::tuple_size_v<Tuple>,
                "type does not appear in tuple");
};

//----------------------------------------------

// Helper for tuple_template_index
template <template <typename TT> typename PropertyContainer, typename Tuple> struct tuple_template_index_helper;

template<template <typename TT> typename PropertyContainer>
struct tuple_template_index_helper<PropertyContainer, std::tuple<>>
{
  static constexpr std::size_t value = 0;
};

template<template <typename TT> typename PropertyContainer, typename T, typename... Rest>
struct tuple_template_index_helper<PropertyContainer, std::tuple<PropertyContainer<T>, Rest...>>
{
  static constexpr std::size_t value = 0;
  using RestTuple = std::tuple<Rest...>;
  static_assert(
    tuple_template_index_helper<PropertyContainer, RestTuple>::value == 
    std::tuple_size_v<RestTuple>,
    "type appears more than once in tuple");
};

template<template <typename TT> typename PropertyContainer, typename First, typename... Rest>
struct tuple_template_index_helper<PropertyContainer, std::tuple<First, Rest...>>
{
  using RestTuple = std::tuple<Rest...>;
  static constexpr std::size_t value = 1 +
       tuple_template_index_helper<PropertyContainer, RestTuple>::value;
};

// tuple_template_index is a trait helper which finds the index of a
// class template in a std::tuple<Ts...>. During template argument deduction
// this enables us to search the tuple for e.g. `kernel_properties` without
// knowing the concrete type (e.g. kernel_properties<KProps>)
// A compile time error is raised if the class template is found more than once.
// If not found, returns the tuple size (i.e. this is not an error).
template<template <typename TT> typename PropertyContainer, typename Tuple>
struct tuple_template_index
{
  static constexpr std::size_t value =
    tuple_template_index_helper<PropertyContainer, Tuple>::value;
};

// tuple_contains_template piggy-backs on the functionality of
// tuple_template_index to detect whether a class template exists in the tuple
template <template <typename TT> typename PropertyContainer, typename Tuple>
    struct tuple_contains_template
    : std::conditional_t <
      tuple_template_index<PropertyContainer, Tuple>::value<
          std::tuple_size_v<Tuple>, std::true_type, std::false_type> {};


template <bool TupleContains, typename PropertyContainerConcrete, typename Tuple>
struct property_getter_helper;

template <typename PropertyContainerConcrete, typename Tuple>
struct property_getter_helper<true, PropertyContainerConcrete, Tuple>{
  PropertyContainerConcrete operator()(Tuple tuple){
    return std::get<PropertyContainerConcrete>(tuple);
  }
};

template <typename PropertyContainerConcrete, typename Tuple>
struct property_getter_helper<false, PropertyContainerConcrete, Tuple>{
  PropertyContainerConcrete operator()(Tuple tuple){
    (void)tuple;
    return {};
  }
};

// For local_mem_size
template <typename T, typename Tuple>
struct has_type;

template <typename T, typename... Us>
struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

template <template <typename TT> typename PropertyContainer,  typename PropertyContainerConcrete, typename Tuple>
using property_getter = property_getter_helper<tuple_contains_template<PropertyContainer, Tuple>::value, PropertyContainerConcrete, Tuple>;

template <typename PropertyContainerConcrete, typename Tuple>
using local_mem_getter = property_getter_helper<has_type<PropertyContainerConcrete, Tuple>::value, PropertyContainerConcrete, Tuple>;



//TODO: ought this to return the sycl::properties type or the wrapper type?
template <bool InTuple, template <typename TT> typename PropertyContainer, typename... Ts>
struct properties_or_empty_helper;

template <template <typename TT> typename PropertyContainer, typename... Ts>
struct properties_or_empty_helper<false, PropertyContainer, Ts...> {
  using Props = sycl::ext::oneapi::experimental::empty_properties_t;
};

template <template <typename TT> typename PropertyContainer, typename... Ts>
struct properties_or_empty_helper<true, PropertyContainer, Ts...> {
  using Props = typename std::tuple_element_t<
      tuple_template_index<PropertyContainer, std::tuple<Ts...>>::value,
      std::tuple<Ts...>>::Props;
};

template <template <typename TT> typename PropertyContainer, typename... Ts>
using properties_or_empty = typename properties_or_empty_helper<tuple_contains_template<PropertyContainer, std::tuple<Ts...>>::value, PropertyContainer, Ts...>::Props;

} // namespace detail

//----------------------------------------------

// launch_policy is constructed by the user & passed to `compat_exp::launch`
template <typename Range, typename KProps, typename LProps, bool LocalMem>
struct launch_policy {
  static_assert(sycl_exp::is_property_list_v<KProps>);
  static_assert(sycl_exp::is_property_list_v<LProps>);
  static_assert(syclcompat::detail::is_range_or_nd_range_v<Range>);
  static_assert(syclcompat::detail::is_nd_range_v<Range> || !LocalMem,
                "\nsycl::range kernel launches are incompatible with local "
                "memory usage!");

  using KPropsT = KProps;
  using LPropsT = LProps;
  using RangeT = Range;
  static constexpr bool HasLocalMem = LocalMem;
  static constexpr int Dim = syclcompat::detail::range_dimension_v<Range>;

private:
  launch_policy() = default;

  template <typename... Ts>
  launch_policy(Ts... ts)
      : _kernel_properties{detail::property_getter<
            kernel_properties, kernel_properties<KPropsT>, std::tuple<Ts...>>()(
            std::tuple<Ts...>(ts...))},
        _launch_properties{detail::property_getter<
            launch_properties, launch_properties<LPropsT>, std::tuple<Ts...>>()(
            std::tuple<Ts...>(ts...))},
        _local_mem_size{detail::local_mem_getter<
            local_mem_size, std::tuple<Ts...>>()(std::tuple<Ts...>(ts...))} {
    static_assert(
        std::conjunction_v<std::disjunction<detail::is_kernel_properties<Ts>,
                                            detail::is_launch_properties<Ts>,
                                            detail::is_local_mem_size<Ts>>...>,
        "\nReceived an unexpected argument to ctor. Did you forget to wrap "
        "in "
        "compat::kernel_properties, launch_properties, local_mem_size?");
  }

public:
  template <typename... Ts>
  launch_policy(Range range, Ts... ts) : launch_policy(ts...) {
    _range = range;
  }

  template <typename... Ts>
  launch_policy(dim3 global_range, Ts... ts) : launch_policy(ts...) {
    _range = Range{global_range};
  }

  template <typename... Ts>
  launch_policy(dim3 global_range, dim3 local_range, Ts... ts) : launch_policy(ts...) {
    _range = Range{global_range, local_range};
  }

  KProps get_kernel_properties() { return _kernel_properties.props; }
  LProps get_launch_properties() { return _launch_properties.props; }
  size_t get_local_mem_size() { return _local_mem_size.size; }
  Range get_range() { return _range; }

private:
  Range _range;
  kernel_properties<KProps> _kernel_properties;
  launch_properties<LProps> _launch_properties;
  local_mem_size _local_mem_size;
};

template <typename Range, typename... Ts>
launch_policy(Range, Ts...)
    -> launch_policy<Range,
                     detail::properties_or_empty<kernel_properties, Ts...>,
                     detail::properties_or_empty<launch_properties, Ts...>,
                     detail::has_type<local_mem_size, std::tuple<Ts...>>::value>;

template <typename... Ts>
launch_policy(dim3, Ts...)
    -> launch_policy<sycl::range<3>,
                     detail::properties_or_empty<kernel_properties, Ts...>,
                     detail::properties_or_empty<launch_properties, Ts...>,
                     detail::has_type<local_mem_size, std::tuple<Ts...>>::value>;

template <typename... Ts>
launch_policy(dim3, dim3, Ts...)
    -> launch_policy<sycl::nd_range<3>,
                     detail::properties_or_empty<kernel_properties, Ts...>,
                     detail::properties_or_empty<launch_properties, Ts...>,
                     detail::has_type<local_mem_size, std::tuple<Ts...>>::value>;

template <typename T> struct is_launch_policy : std::false_type {};

template <typename RangeT, typename KProps, typename LProps, bool LocalMem>
struct is_launch_policy<launch_policy<RangeT, KProps, LProps, LocalMem>> : std::true_type {};

template <typename T>
inline constexpr bool is_launch_policy_v = is_launch_policy<T>::value;


namespace detail {

template <auto F, typename Range, typename KProps, bool HasLocalMem, typename... Args> struct KernelFunctor {
  KernelFunctor(KProps kernel_props, Args... args)
      : _kernel_properties{kernel_props},
        _argument_tuple(std::make_tuple(args...)) {}

  KernelFunctor(KProps kernel_props, sycl::local_accessor<char, 1> local_acc,
                Args... args)
      : _kernel_properties{kernel_props}, _local_acc{local_acc},
        _argument_tuple(std::make_tuple(args...)) {}

  auto get(sycl_exp::properties_tag) { return _kernel_properties; }

  __syclcompat_inline__ inline void
  operator()(syclcompat::detail::range_to_item_t<Range> it) const {
    if constexpr (HasLocalMem) {
      char *local_mem_ptr = static_cast<char *>(
          _local_acc.template get_multi_ptr<sycl::access::decorated::no>());
      std::apply(
          [lmem_ptr = local_mem_ptr](auto &&...args) { F(args..., lmem_ptr); },
          _argument_tuple);
    } else {
      std::apply([](auto &&...args) { F(args...); }, _argument_tuple);
    }
  }

  KProps _kernel_properties;
  std::tuple<Args...> _argument_tuple;
  std::conditional_t<HasLocalMem, sycl::local_accessor<char, 1>, std::monostate>
      _local_acc; //monostate for empty type
};

//====================================================================
// This helper function avoids 2 nested `if constexpr` in detail::launch
template <auto F, typename LaunchPolicy, typename... Args>
auto build_kernel_functor(sycl::handler& cgh, LaunchPolicy launch_policy,
                          Args... args)
    -> KernelFunctor<F, typename LaunchPolicy::RangeT,
                     typename LaunchPolicy::KPropsT, LaunchPolicy::HasLocalMem, Args...> {
  if constexpr (LaunchPolicy::HasLocalMem) {
    sycl::local_accessor<char, 1> local_memory(launch_policy.get_local_mem_size(),
                                               cgh);
    return KernelFunctor<F, typename LaunchPolicy::RangeT,
                         typename LaunchPolicy::KPropsT, LaunchPolicy::HasLocalMem, Args...>(
        launch_policy.get_kernel_properties(), local_memory, args...);
  } else {
      return KernelFunctor<F, typename LaunchPolicy::RangeT,
                        typename LaunchPolicy::KPropsT, LaunchPolicy::HasLocalMem, Args...>(
              launch_policy.get_kernel_properties(), args...);
  }
}

template <auto F, typename LaunchPolicy, typename... Args>
sycl::event launch(LaunchPolicy launch_policy, sycl::queue q, Args... args) {
  static_assert(syclcompat::args_compatible<LaunchPolicy, F, Args...>,
                "Mismatch between device function signature and supplied "
                "arguments. Have you correctly handled local memory/char*?");

  sycl_exp::launch_config config(launch_policy.get_range(),
                                 launch_policy.get_launch_properties());

  return sycl_exp::submit_with_event(q, [&](sycl::handler &cgh) {
    auto KernelFunctor = build_kernel_functor<F>(cgh, launch_policy, args...);
    if constexpr (syclcompat::detail::is_range_v<
                      typename LaunchPolicy::RangeT>) {
      parallel_for(cgh, config, KernelFunctor);
    } else {
      static_assert(syclcompat::detail::is_nd_range_v<
                    typename LaunchPolicy::RangeT>);
      nd_launch(cgh, config, KernelFunctor);
    }
  });
}

} // namespace detail

template <auto F, typename LaunchPolicy, typename... Args>
sycl::event launch(LaunchPolicy launch_policy, sycl::queue q, Args... args) {
  static_assert(is_launch_policy_v<LaunchPolicy>);
  return detail::launch<F>(launch_policy, q, args...);
}

template <auto F, typename LaunchPolicy, typename... Args>
sycl::event launch(LaunchPolicy launch_policy, Args... args) {
  static_assert(is_launch_policy_v<LaunchPolicy>);
  return launch<F>(launch_policy, get_default_queue(), args...);
}

} // namespace experimental
} // namespace syclcompat
