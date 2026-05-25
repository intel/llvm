//===------------------ NegativeConversions.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

template <typename ElementType, address_space Space, decorated IsDecorated>
using multi_ptr_t = sycl::multi_ptr<ElementType, Space, IsDecorated>;

TEST(MultiPtrNegativeConversions, CannotRemoveConstQualification) {
  using const_ptr =
      multi_ptr_t<const int, address_space::private_space, decorated::no>;
  using mutable_ptr =
      multi_ptr_t<int, address_space::private_space, decorated::no>;

  EXPECT_FALSE((std::is_constructible_v<mutable_ptr, const_ptr>));
  EXPECT_FALSE((std::is_convertible_v<const_ptr, mutable_ptr>));
  EXPECT_FALSE((std::is_assignable_v<mutable_ptr, const_ptr>));
}

TEST(MultiPtrNegativeConversions, CannotConvertBetweenIncompatibleTypes) {
  using int_ptr = multi_ptr_t<int, address_space::private_space, decorated::no>;
  using float_ptr =
      multi_ptr_t<float, address_space::private_space, decorated::no>;
  using double_ptr =
      multi_ptr_t<double, address_space::private_space, decorated::no>;

  EXPECT_FALSE((std::is_constructible_v<int_ptr, float_ptr>));
  EXPECT_FALSE((std::is_constructible_v<float_ptr, int_ptr>));
  EXPECT_FALSE((std::is_constructible_v<int_ptr, double_ptr>));
  EXPECT_FALSE((std::is_convertible_v<int_ptr, float_ptr>));
  EXPECT_FALSE((std::is_convertible_v<float_ptr, double_ptr>));
}

TEST(MultiPtrNegativeConversions, CannotConvertToGenericFromNonGeneric) {
  using private_ptr =
      multi_ptr_t<int, address_space::private_space, decorated::no>;
  using local_ptr = multi_ptr_t<int, address_space::local_space, decorated::no>;
  using global_ptr =
      multi_ptr_t<int, address_space::global_space, decorated::no>;
  using generic_ptr =
      multi_ptr_t<int, address_space::generic_space, decorated::no>;

  EXPECT_FALSE((std::is_convertible_v<private_ptr, generic_ptr>));
  EXPECT_FALSE((std::is_convertible_v<local_ptr, generic_ptr>));
  EXPECT_FALSE((std::is_convertible_v<global_ptr, generic_ptr>));
}

TEST(MultiPtrNegativeConversions,
     CannotImplicitlyConvertFromGenericToSpecificSpace) {
  using generic_ptr =
      multi_ptr_t<int, address_space::generic_space, decorated::no>;
  using private_ptr =
      multi_ptr_t<int, address_space::private_space, decorated::no>;
  using local_ptr = multi_ptr_t<int, address_space::local_space, decorated::no>;
  using global_ptr =
      multi_ptr_t<int, address_space::global_space, decorated::no>;

  EXPECT_FALSE((std::is_convertible_v<generic_ptr, private_ptr>));
  EXPECT_FALSE((std::is_convertible_v<generic_ptr, local_ptr>));
  EXPECT_FALSE((std::is_convertible_v<generic_ptr, global_ptr>));
}

TEST(MultiPtrNegativeConversions, CannotConvertGenericToConstantSpace) {
  using generic_ptr =
      multi_ptr_t<const int, address_space::generic_space, decorated::no>;
  using constant_ptr =
      multi_ptr_t<const int, address_space::constant_space, decorated::no>;

  generic_ptr gen_ptr;
  constant_ptr const_ptr;

  generic_ptr gen_ptr2{const_ptr};
  constant_ptr const_ptr2{gen_ptr};

  EXPECT_EQ(gen_ptr2.get(), const_ptr.get());
  EXPECT_EQ(const_ptr2.get(), gen_ptr.get());
}

TEST(MultiPtrNegativeConversions, CannotConvertVoidToTypedWithoutExplicitCast) {
  using void_ptr =
      multi_ptr_t<void, address_space::private_space, decorated::no>;
  using int_ptr = multi_ptr_t<int, address_space::private_space, decorated::no>;

  EXPECT_FALSE((std::is_convertible_v<void_ptr, int_ptr>));
}

TEST(MultiPtrNegativeConversions, CannotConstructMutableFromConstTypedVoid) {
  using const_void_ptr =
      multi_ptr_t<const void, address_space::private_space, decorated::no>;
  using int_ptr = multi_ptr_t<int, address_space::private_space, decorated::no>;

  EXPECT_FALSE((std::is_constructible_v<int_ptr, const_void_ptr>));
}

TEST(MultiPtrNegativeConversions, CannotAssignBetweenIncompatibleSpaces) {
  using private_ptr =
      multi_ptr_t<int, address_space::private_space, decorated::no>;
  using local_ptr = multi_ptr_t<int, address_space::local_space, decorated::no>;

  EXPECT_FALSE((std::is_assignable_v<private_ptr, local_ptr>));
  EXPECT_FALSE((std::is_assignable_v<local_ptr, private_ptr>));
}

TEST(MultiPtrNegativeConversions, GenericCannotAssignFromConstantSpace) {
  using generic_ptr =
      multi_ptr_t<const int, address_space::generic_space, decorated::no>;
  using constant_ptr =
      multi_ptr_t<const int, address_space::constant_space, decorated::no>;

  EXPECT_FALSE((std::is_assignable_v<generic_ptr, constant_ptr>));
}

TEST(MultiPtrNegativeConversions, ConstantSpaceNotSupportedForVoidInNonLegacy) {
  using const_void_constant_legacy =
      multi_ptr_t<const void, address_space::constant_space, decorated::legacy>;
  EXPECT_TRUE((std::is_default_constructible_v<const_void_constant_legacy>));
}

TEST(MultiPtrNegativeConversions, CannotComparePointersOfDifferentTypes) {
  using int_ptr = multi_ptr_t<int, address_space::private_space, decorated::no>;
  using float_ptr =
      multi_ptr_t<float, address_space::private_space, decorated::no>;

  EXPECT_FALSE((std::is_invocable_v<std::equal_to<>, int_ptr, float_ptr>));
}

} // namespace
