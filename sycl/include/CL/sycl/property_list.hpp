//==--------- property_list.hpp --- SYCL property list ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_list_base.hpp>
#include <CL/sycl/properties/property_traits.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
template <typename... PropsT> class accessor_property_list;
}
} // namespace ext

/// Objects of the property_list class are containers for the SYCL properties
///
/// \ingroup sycl_api
class property_list : protected detail::PropertyListBase {

  // The structs validate that all objects passed are SYCL properties
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : detail::conditional_t<is_property<T>::value, AllProperties<Tail...>,
                              std::false_type> {};

public:
  template <typename... PropsT, typename = typename detail::enable_if_t<
                                    AllProperties<PropsT...>::value>>
  property_list(PropsT... Props) : detail::PropertyListBase(false) {
    ctorHelper(Props...);
  }

  template <typename PropT> PropT get_property() const {
    if (!has_property<PropT>())
      throw sycl::invalid_object_error("The property is not found",
                                       PI_INVALID_VALUE);

    return get_property_helper<PropT>();
  }

  template <typename PropT> bool has_property() const {
    return has_property_helper<PropT>();
  }

  void add_or_replace_accessor_properties(const property_list &PropertyList) {
    add_or_replace_accessor_properties_helper(PropertyList.MPropsWithData);
  }
  void delete_accessor_property(const sycl::detail::PropWithDataKind &Kind) {
    delete_accessor_property_helper(Kind);
  }

  template <typename... T> operator ext::oneapi::accessor_property_list<T...>();

private:
  template <typename... PropsT>
  friend class ext::oneapi::accessor_property_list;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
