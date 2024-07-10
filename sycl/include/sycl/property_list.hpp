//==--------- property_list.hpp --- SYCL property list ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/pi.h>                    // for PI_ERROR_INVALID_VALUE
#include <sycl/detail/property_helper.hpp>     // for DataLessPropKind, Pro...
#include <sycl/detail/property_list_base.hpp>  // for PropertyListBase
#include <sycl/exception.hpp>
#include <sycl/properties/property_traits.hpp> // for is_property

#include <bitset>      // for bitset
#include <memory>      // for shared_ptr
#include <type_traits> // for conditional_t, enable...
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
template <typename... PropsT> class accessor_property_list;
} // namespace ext::oneapi

/// Objects of the property_list class are containers for the SYCL properties
///
/// \ingroup sycl_api
class property_list : protected detail::PropertyListBase {

  // The structs validate that all objects passed are SYCL properties
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : std::conditional_t<is_property<T>::value, AllProperties<Tail...>,
                           std::false_type> {};

public:
  template <typename... PropsT, typename = typename std::enable_if_t<
                                    AllProperties<PropsT...>::value>>
  property_list(PropsT... Props) : detail::PropertyListBase(false) {
    ctorHelper(Props...);
  }

  template <typename PropT> PropT get_property() const {
    if (!has_property<PropT>())
      throw sycl::exception(make_error_code(errc::invalid),
                            "The property is not found");

    return get_property_helper<PropT>();
  }

  template <typename PropT> bool has_property() const noexcept {
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
  property_list(
      std::bitset<detail::DataLessPropKind::DataLessPropKindSize> DataLessProps,
      std::vector<std::shared_ptr<detail::PropertyWithDataBase>> PropsWithData)
      : sycl::detail::PropertyListBase(DataLessProps, PropsWithData) {}

  template <typename... PropsT>
  friend class ext::oneapi::accessor_property_list;
};

} // namespace _V1
} // namespace sycl
