//==----- accessor_property_list.hpp --- SYCL accessor property list -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_list_base.hpp>
#include <CL/sycl/property_list.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {} // namespace detail
namespace ext {
namespace ONEAPI {

template <typename T> struct is_property : std::false_type {};

template <typename T> struct is_compile_time_property : std::false_type {};

/// Objects of the accessor_property_list class are containers for the SYCL
/// properties.
///
/// Unlike \c property_list, accessor_property_list can take
/// compile-time-constant properties.
///
/// \sa accessor
/// \sa property_list
///
/// \ingroup sycl_api
template <typename... PropsT>
class accessor_property_list : protected detail::PropertyListBase {
  // These structures check if compile-time-constant property is present in
  // list. For runtime properties this check is always true.
  template <class T, class U> struct AreSameTemplate : std::is_same<T, U> {};
  template <template <class...> class T, class T1, class T2>
  struct AreSameTemplate<T<T1>, T<T2>> : std::true_type {};
#if __cplusplus >= 201703L
  template <template <auto...> class T, auto...T1, auto...T2>
  struct AreSameTemplate<T<T1...>, T<T2...>> : std::true_type {};
#endif
  template <typename PropT, typename... PropListT>
  struct ContainsProperty;
  template <typename PropT> struct ContainsProperty<PropT> : std::false_type {};
  template <typename PropT, typename Head, typename... Tail>
  struct ContainsProperty<PropT, Head, Tail...>
      : std::conditional<AreSameTemplate<PropT, Head>::value ||
                             !is_compile_time_property<PropT>::value,
                         std::true_type,
                         ContainsProperty<PropT, Tail...>>::type {};

  // The following structures help to check if two property lists contain the
  // same compile-time-constant properties.
  template <typename...> struct PropertyContainer {
    using Head = void;
    using Rest = void;
  };
  template <typename T, typename... Other>
  struct PropertyContainer<T, Other...> {
    using Head = T;
    using Rest = PropertyContainer<Other...>;
  };
  template <typename T> struct PropertyContainer<T> {
    using Head = T;
    using Rest = void;
  };

  template <typename ContainerT, typename... OtherProps>
  struct ContainsSameProperties
      : std::conditional<
            ContainsProperty<typename ContainerT::Head, OtherProps...>::value,
            ContainsSameProperties<typename ContainerT::Rest, OtherProps...>,
            std::false_type>::type {};
  template <typename... OtherProps>
  struct ContainsSameProperties<void, OtherProps...> : std::true_type {};

#if __cplusplus >= 201703L
  template <typename ContainerT, template <auto...> class PropT, auto... Args>
  struct GetCxPropertyHelper {
    using type = typename std::conditional_t<
        AreSameTemplate<typename ContainerT::Head, PropT<Args...>>::value,
        typename ContainerT::Head,
        typename GetCxPropertyHelper<typename ContainerT::Rest, PropT,
                                     Args...>::type>;
  };
  template <typename Head, template <auto...> class PropT, auto... Args>
  struct GetCxPropertyHelper<PropertyContainer<Head>, PropT, Args...> {
    using type = typename std::conditional_t<
        AreSameTemplate<Head, PropT<Args...>>::value, Head, void>;
  };
#endif

  // The structs validate that all objects passed are SYCL properties
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : std::conditional<
            std::is_base_of<detail::DataLessPropertyBase, T>::value ||
                std::is_base_of<detail::PropertyWithDataBase, T>::value ||
                is_compile_time_property<T>::value,
            AllProperties<Tail...>, std::false_type>::type {};

public:
  template <
      typename = typename std::enable_if<AllProperties<PropsT...>::value>::type>
  accessor_property_list(PropsT... Props) : detail::PropertyListBase(false) {
    ctorHelper(Props...);
  }

  template <typename... OtherProps,
            typename = typename std::enable_if<
                ContainsSameProperties<PropertyContainer<PropsT...>,
                                       OtherProps...>::value &&
                ContainsSameProperties<PropertyContainer<OtherProps...>,
                                       PropsT...>::value>::type>
  accessor_property_list(const accessor_property_list<OtherProps...> &OtherList)
      : detail::PropertyListBase(OtherList.MDataLessProps,
                                 OtherList.MPropsWithData) {}

  template <typename PropT,
            typename = std::enable_if<!is_compile_time_property<PropT>::value>>
  PropT get_property() const {
    if (!has_property<PropT>())
      throw sycl::invalid_object_error("The property is not found",
                                       PI_INVALID_VALUE);

    return get_property_helper<PropT>();
  }

  template <class PropT>
  typename std::enable_if<!is_compile_time_property<PropT>::value, bool>::type
  has_property() const {
    return has_property_helper<PropT>();
  }

#if __cplusplus >= 201703L
  template <template <auto...> class PropT, auto... Args>
  static constexpr
      typename std::enable_if<is_compile_time_property<PropT<Args...>>::value,
                              bool>::type
      has_property() {
    return ContainsProperty<PropT<Args...>, PropsT...>::value;
  }
  template <template <auto...> class PropT, auto... Args,
            typename = typename std::enable_if<
                is_compile_time_property<PropT<Args...>>::value>::type>
  static constexpr auto get_property() {
    return typename GetCxPropertyHelper<PropertyContainer<PropsT...>, PropT,
                                        Args...>::type{};
  }
#endif

private:
  template <typename... OtherProps> friend class accessor_property_list;
  template <typename, int, sycl::access::mode, sycl::access::target,
            sycl::access::placeholder, typename>
  friend class accessor;

public:
  template <typename... OtherPropsT>
  static constexpr bool areSameCxProperties() {
    return ContainsSameProperties<PropertyContainer<OtherPropsT...>,
                                  PropsT...>::value;
  }
};
} // namespace ONEAPI
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
