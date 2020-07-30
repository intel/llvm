//==--------- property_list.hpp --- SYCL property list ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/property_helper.hpp>

#include <bitset>
#include <memory>
#include <type_traits>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

/// Objects of the property_list class are containers for the SYCL properties
///
/// \ingroup sycl_api
class property_list {

  // The structs validate that all objects passed are SYCL properties
  template <typename... Tail> struct AllProperties : std::true_type {};
  template <typename T, typename... Tail>
  struct AllProperties<T, Tail...>
      : std::conditional<
            std::is_base_of<detail::DataLessPropertyBase, T>::value ||
                std::is_base_of<detail::PropertyWithDataBase, T>::value,
            AllProperties<Tail...>, std::false_type>::type {};

public:
  template <typename... PropsT, typename = typename std::enable_if<
                                    AllProperties<PropsT...>::value>::type>
  property_list(PropsT... Props) : MDataLessProps(false) {
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

private:
  void ctorHelper() {}

  template <typename... PropsT, class PropT>
  typename std::enable_if<
      std::is_base_of<detail::DataLessPropertyBase, PropT>::value>::type
  ctorHelper(PropT &, PropsT... Props) {
    const int PropKind = static_cast<int>(PropT::getKind());
    MDataLessProps[PropKind] = true;
    ctorHelper(Props...);
  }

  template <typename... PropsT, class PropT>
  typename std::enable_if<
      std::is_base_of<detail::PropertyWithDataBase, PropT>::value>::type
  ctorHelper(PropT &Prop, PropsT... Props) {
    MPropsWithData.emplace_back(new PropT(Prop));
    ctorHelper(Props...);
  }

  template <typename PropT>
  typename std::enable_if<
      std::is_base_of<detail::DataLessPropertyBase, PropT>::value, bool>::type
  has_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    if (PropKind >= detail::DataLessPropKind::DataLessPropKindSize)
      return false;
    return MDataLessProps[PropKind];
  }

  template <typename PropT>
  typename std::enable_if<
      std::is_base_of<detail::PropertyWithDataBase, PropT>::value, bool>::type
  has_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    for (const std::shared_ptr<detail::PropertyWithDataBase> &Prop :
         MPropsWithData)
      if (Prop->isSame(PropKind))
        return true;
    return false;
  }

  template <typename PropT>
  typename std::enable_if<
      std::is_base_of<detail::DataLessPropertyBase, PropT>::value, PropT>::type
  get_property_helper() const {
    // In case of simple property we can just construct it
    return PropT{};
  }

  template <typename PropT>
  typename std::enable_if<
      std::is_base_of<detail::PropertyWithDataBase, PropT>::value, PropT>::type
  get_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    if (PropKind >= detail::PropWithDataKind::PropWithDataKindSize)
      throw sycl::invalid_object_error("The property is not found",
                                       PI_INVALID_VALUE);

    for (const std::shared_ptr<detail::PropertyWithDataBase> &Prop :
         MPropsWithData)
      if (Prop->isSame(PropKind))
        return *static_cast<PropT *>(Prop.get());

    throw sycl::invalid_object_error("The property is not found",
                                     PI_INVALID_VALUE);
  }

private:
  // Stores enable/not enabled for simple properties
  std::bitset<detail::DataLessPropKind::DataLessPropKindSize> MDataLessProps;
  // Stores shared_ptrs to complex properties
  std::vector<std::shared_ptr<detail::PropertyWithDataBase>> MPropsWithData;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
