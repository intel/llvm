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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declaration
template <typename, int, access::mode, access::target, access::placeholder,
          typename PropertyListT>
class accessor;
namespace detail {
// This helper template must be specialized for nested instance template
// of each compile-time-constant property.
template <typename T> struct IsCompileTimePropertyInstance : std::false_type {};
} // namespace detail
namespace ext {
namespace oneapi {

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
class accessor_property_list : protected sycl::detail::PropertyListBase {
  // These structures check if compile-time-constant property is present in
  // list. For runtime properties this check is always true.
  template <class T, class U> struct AreSameTemplate : std::is_same<T, U> {};
  template <template <class...> class T, class T1, class T2>
  struct AreSameTemplate<T<T1>, T<T2>> : std::true_type {};
#if __cplusplus >= 201703L
  // Declaring non-type template parameters with auto is a C++17 feature. Since
  // the extension is written against SYCL 2020, which implies use of C++17,
  // there's no need to provide alternative implementations for older standards.
  template <template <auto...> class T, auto... T1, auto... T2>
  struct AreSameTemplate<T<T1...>, T<T2...>> : std::true_type {};
#endif
  // This template helps to identify if PropListT parameter pack contains
  // property of PropT type, where PropT is a nested instance template of
  // compile-time-constant property.
  template <typename PropT, typename... PropListT> struct ContainsProperty;
  template <typename PropT> struct ContainsProperty<PropT> : std::false_type {};
  template <typename PropT, typename Head, typename... Tail>
  struct ContainsProperty<PropT, Head, Tail...>
      : sycl::detail::conditional_t<AreSameTemplate<PropT, Head>::value,
                                    std::true_type,
                                    ContainsProperty<PropT, Tail...>> {};

  // PropertyContainer is a helper structure, that holds list of properties.
  // It is used to avoid multiple parameter packs in templates.
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

#if __cplusplus >= 201703L
  // This template serves the same purpose as ContainsProperty, but operates on
  // template template arguments.
  template <typename ContainerT, template <auto...> typename PropT,
            auto... Args>
  struct ContainsPropertyInstance
      : sycl::detail::conditional_t<
            !std::is_same_v<typename ContainerT::Head, void> &&
                AreSameTemplate<PropT<Args...>,
                                typename ContainerT::Head>::value,
            std::true_type,
            ContainsPropertyInstance<typename ContainerT::Rest, PropT,
                                     Args...>> {};

  template <template <auto...> typename PropT, auto... Args>
  struct ContainsPropertyInstance<void, PropT, Args...> : std::false_type {};
#endif

  // This template checks if two lists of properties contain the same set of
  // compile-time-constant properties in any order. Run time properties are
  // skipped.
  template <typename ContainerT, typename... OtherProps>
  struct ContainsSameProperties
      : sycl::detail::conditional_t<
            !sycl::detail::IsCompileTimePropertyInstance<
                typename ContainerT::Head>::value ||
                ContainsProperty<typename ContainerT::Head,
                                 OtherProps...>::value,
            ContainsSameProperties<typename ContainerT::Rest, OtherProps...>,
            std::false_type> {};
  template <typename... OtherProps>
  struct ContainsSameProperties<void, OtherProps...> : std::true_type {};

#if __cplusplus >= 201703L
  // This template helps to extract exact property instance type based on
  // template template argument. If there's an instance of target property in
  // ContainerT, find instance template and use it as type. Otherwise, just
  // use void as return type.
  template <typename ContainerT, template <auto...> class PropT, auto... Args>
  struct GetCompileTimePropertyHelper {
    using type = typename sycl::detail::conditional_t<
        AreSameTemplate<typename ContainerT::Head, PropT<Args...>>::value,
        typename ContainerT::Head,
        typename GetCompileTimePropertyHelper<typename ContainerT::Rest, PropT,
                                              Args...>::type>;
  };
  template <typename Head, template <auto...> class PropT, auto... Args>
  struct GetCompileTimePropertyHelper<PropertyContainer<Head>, PropT, Args...> {
    using type = typename sycl::detail::conditional_t<
        AreSameTemplate<Head, PropT<Args...>>::value, Head, void>;
  };
#endif

  // The structs validate that all objects passed are SYCL properties.
  // Properties are either run time SYCL 1.2.1 properties, and thus derive from
  // either DataLessPropertyBase or from PropertyWithDataBase, or
  // compile-time-constant properties, and thus specialize
  // IsCompileTimePropertyInstance template.
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : sycl::detail::conditional_t<
            std::is_base_of<sycl::detail::DataLessPropertyBase, T>::value ||
                std::is_base_of<sycl::detail::PropertyWithDataBase, T>::value ||
                sycl::detail::IsCompileTimePropertyInstance<T>::value,
            AllProperties<Tail...>, std::false_type> {};

  accessor_property_list(
      std::bitset<sycl::detail::DataLessPropKind::DataLessPropKindSize>
          DataLessProps,
      std::vector<std::shared_ptr<sycl::detail::PropertyWithDataBase>>
          PropsWithData)
      : sycl::detail::PropertyListBase(DataLessProps, PropsWithData) {}

public:
  template <typename = typename sycl::detail::enable_if_t<
                AllProperties<PropsT...>::value>>
  accessor_property_list(PropsT... Props)
      : sycl::detail::PropertyListBase(false) {
    ctorHelper(Props...);
  }

  accessor_property_list(const sycl::property_list &Props)
      : sycl::detail::PropertyListBase(Props.MDataLessProps,
                                       Props.MPropsWithData) {}

  template <typename... OtherProps,
            typename = typename sycl::detail::enable_if_t<
                ContainsSameProperties<PropertyContainer<PropsT...>,
                                       OtherProps...>::value &&
                ContainsSameProperties<PropertyContainer<OtherProps...>,
                                       PropsT...>::value>>
  accessor_property_list(const accessor_property_list<OtherProps...> &OtherList)
      : sycl::detail::PropertyListBase(OtherList.MDataLessProps,
                                       OtherList.MPropsWithData) {}

  template <typename PropT, typename = typename sycl::detail::enable_if_t<
                                !is_compile_time_property<PropT>::value>>
  PropT get_property() const {
    if (!has_property<PropT>())
      throw sycl::invalid_object_error("The property is not found",
                                       PI_INVALID_VALUE);

    return get_property_helper<PropT>();
  }

  template <class PropT>
  typename sycl::detail::enable_if_t<!is_compile_time_property<PropT>::value,
                                     bool>
  has_property() const {
    return has_property_helper<PropT>();
  }

#if __cplusplus >= 201703L
  template <typename T>
  static constexpr bool has_property(
      typename std::enable_if_t<is_compile_time_property<T>::value> * = 0) {
    return ContainsPropertyInstance<PropertyContainer<PropsT...>,
                                    T::template instance>::value;
  }

  template <typename T>
  static constexpr auto get_property(
      typename std::enable_if_t<
          is_compile_time_property<T>::value &&
          ContainsPropertyInstance<PropertyContainer<PropsT...>,
                                   T::template instance>::value> * = 0) {
    return typename GetCompileTimePropertyHelper<PropertyContainer<PropsT...>,
                                                 T::template instance>::type{};
  }
#endif

private:
  template <typename, int, access::mode, access::target, access::placeholder,
            typename PropertyListT>
  friend class sycl::accessor;

  template <typename... OtherProps> friend class accessor_property_list;

  friend class sycl::property_list;

  // Helper method, used by accessor to restrict conversions to compatible
  // property lists.
  template <typename... OtherPropsT>
  static constexpr bool areSameCompileTimeProperties() {
    return ContainsSameProperties<PropertyContainer<OtherPropsT...>,
                                  PropsT...>::value;
  }
};
} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
