//==------- property_list_base.hpp --- Base for SYCL property lists --------==//
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
namespace detail {
class PropertyListBase {
protected:
  explicit PropertyListBase(
      std::bitset<DataLessPropKind::DataLessPropKindSize> DataLessProps)
      : MDataLessProps(DataLessProps) {}
  PropertyListBase(
      std::bitset<DataLessPropKind::DataLessPropKindSize> DataLessProps,
      std::vector<std::shared_ptr<PropertyWithDataBase>> PropsWithData)
      : MDataLessProps(DataLessProps),
        MPropsWithData(std::move(PropsWithData)) {}
  void ctorHelper() {}

  template <typename... PropsT, class PropT>
  typename std::enable_if<
      std::is_base_of<DataLessPropertyBase, PropT>::value>::type
  ctorHelper(PropT &, PropsT... Props) {
    const int PropKind = static_cast<int>(PropT::getKind());
    MDataLessProps[PropKind] = true;
    ctorHelper(Props...);
  }

  template <typename... PropsT, class PropT>
  typename std::enable_if<
      std::is_base_of<PropertyWithDataBase, PropT>::value>::type
  ctorHelper(PropT &Prop, PropsT... Props) {
    MPropsWithData.emplace_back(new PropT(Prop));
    ctorHelper(Props...);
  }

  // Compile-time-constant properties are simply skipped
  template <typename... PropsT, class PropT>
  typename std::enable_if<
      !std::is_base_of<PropertyWithDataBase, PropT>::value &&
      !std::is_base_of<DataLessPropertyBase, PropT>::value>::type
  ctorHelper(PropT &, PropsT... Props) {
    ctorHelper(Props...);
  }

  template <typename PropT>
  typename std::enable_if<std::is_base_of<DataLessPropertyBase, PropT>::value,
                          bool>::type
  has_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    if (PropKind >= detail::DataLessPropKind::DataLessPropKindSize)
      return false;
    return MDataLessProps[PropKind];
  }

  template <typename PropT>
  typename std::enable_if<std::is_base_of<PropertyWithDataBase, PropT>::value,
                          bool>::type
  has_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    for (const std::shared_ptr<PropertyWithDataBase> &Prop : MPropsWithData)
      if (Prop->isSame(PropKind))
        return true;
    return false;
  }

  template <typename PropT>
  typename std::enable_if<std::is_base_of<DataLessPropertyBase, PropT>::value,
                          PropT>::type
  get_property_helper() const {
    // In case of simple property we can just construct it
    return PropT{};
  }

  template <typename PropT>
  typename std::enable_if<std::is_base_of<PropertyWithDataBase, PropT>::value,
                          PropT>::type
  get_property_helper() const {
    const int PropKind = static_cast<int>(PropT::getKind());
    if (PropKind >= PropWithDataKind::PropWithDataKindSize)
      throw sycl::invalid_object_error("The property is not found",
                                       PI_INVALID_VALUE);

    for (const std::shared_ptr<PropertyWithDataBase> &Prop : MPropsWithData)
      if (Prop->isSame(PropKind))
        return *static_cast<PropT *>(Prop.get());

    throw sycl::invalid_object_error("The property is not found",
                                     PI_INVALID_VALUE);
  }

  // Stores enabled/disabled for simple properties
  std::bitset<DataLessPropKind::DataLessPropKindSize> MDataLessProps;
  // Stores shared_ptrs to complex properties
  std::vector<std::shared_ptr<PropertyWithDataBase>> MPropsWithData;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
