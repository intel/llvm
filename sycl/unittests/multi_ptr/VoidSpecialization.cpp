//===------------------- VoidSpecialization.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/multi_ptr.hpp>
#include <utility>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

template <typename ElementType, address_space Space, decorated IsDecorated>
using multi_ptr_t = sycl::multi_ptr<ElementType, Space, IsDecorated>;

template <typename ElementType, address_space Space, decorated IsDecorated>
multi_ptr_t<ElementType, Space, IsDecorated>
makePtr(std::add_pointer_t<ElementType> Ptr) {
  if constexpr (IsDecorated == decorated::legacy) {
    return multi_ptr_t<ElementType, Space, IsDecorated>{Ptr};
  } else {
    return sycl::address_space_cast<Space, IsDecorated>(Ptr);
  }
}

struct TestData {
  int Value;
  float Score;
};

template <decorated IsDecorated> void checkVoidPointerConstruction() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;
  using const_void_ptr =
      multi_ptr_t<const void, address_space::private_space, IsDecorated>;

  void_ptr DefaultVoid;
  const_void_ptr DefaultConstVoid;
  EXPECT_EQ(DefaultVoid, nullptr);
  EXPECT_EQ(DefaultConstVoid, nullptr);

  void_ptr NullVoid{nullptr};
  const_void_ptr NullConstVoid{nullptr};
  EXPECT_EQ(NullVoid, nullptr);
  EXPECT_EQ(NullConstVoid, nullptr);

  TestData Data{42, 3.14f};
  auto TypedPtr =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data);

  void_ptr VoidFromTyped = TypedPtr;
  EXPECT_NE(VoidFromTyped, nullptr);
  // Convert back to typed to verify it points to the right data
  auto TypedCheck = static_cast<
      multi_ptr_t<TestData, address_space::private_space, IsDecorated>>(
      VoidFromTyped);
  EXPECT_EQ(TypedCheck->Value, 42);

  // For const void, need to go through const typed pointer first
  auto ConstTypedPtr = static_cast<
      multi_ptr_t<const TestData, address_space::private_space, IsDecorated>>(
      TypedPtr);
  const_void_ptr ConstVoidFromTyped = ConstTypedPtr;
  EXPECT_NE(ConstVoidFromTyped, nullptr);
  // Convert back to const typed to verify
  auto ConstTypedCheck = static_cast<
      multi_ptr_t<const TestData, address_space::private_space, IsDecorated>>(
      ConstVoidFromTyped);
  EXPECT_EQ(ConstTypedCheck->Value, 42);
}

template <decorated IsDecorated> void checkVoidPointerCopyAndMove() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;

  TestData Data{17, 2.71f};
  auto TypedPtr =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data);
  void_ptr VoidPtr = TypedPtr;

  void_ptr CopiedVoid{VoidPtr};
  EXPECT_EQ(CopiedVoid, VoidPtr);
  // Verify by converting back to typed
  auto TypedCheck = static_cast<
      multi_ptr_t<TestData, address_space::private_space, IsDecorated>>(
      CopiedVoid);
  EXPECT_EQ(TypedCheck->Value, 17);

  void_ptr MovedVoid{std::move(VoidPtr)};
  auto TypedCheck2 = static_cast<
      multi_ptr_t<TestData, address_space::private_space, IsDecorated>>(
      MovedVoid);
  EXPECT_EQ(TypedCheck2->Value, 17);
}

template <decorated IsDecorated> void checkVoidPointerAssignment() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;

  TestData Data1{10, 1.0f};
  TestData Data2{20, 2.0f};

  auto TypedPtr1 =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data1);
  auto TypedPtr2 =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data2);

  void_ptr VoidPtr1 = TypedPtr1;
  void_ptr VoidPtr2 = TypedPtr2;

  EXPECT_NE(VoidPtr1, VoidPtr2);

  VoidPtr1 = VoidPtr2;
  EXPECT_EQ(VoidPtr1, VoidPtr2);
  // Verify by converting back
  auto TypedCheck = static_cast<
      multi_ptr_t<TestData, address_space::private_space, IsDecorated>>(
      VoidPtr1);
  EXPECT_EQ(TypedCheck->Value, 20);

  VoidPtr1 = nullptr;
  EXPECT_EQ(VoidPtr1, nullptr);
  EXPECT_NE(VoidPtr2, nullptr);
}

template <decorated IsDecorated>
void checkVoidPointerExplicitConversionToTyped() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;
  using int_ptr = multi_ptr_t<int, address_space::private_space, IsDecorated>;
  using data_ptr =
      multi_ptr_t<TestData, address_space::private_space, IsDecorated>;

  TestData Data{99, 9.9f};
  auto TypedPtr =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data);
  void_ptr VoidPtr = TypedPtr;

  data_ptr RoundTrip = static_cast<data_ptr>(VoidPtr);
  EXPECT_EQ(RoundTrip->Value, 99);
  EXPECT_EQ(RoundTrip->Score, 9.9f);

  int Value = 123;
  auto IntTypedPtr =
      makePtr<int, address_space::private_space, IsDecorated>(&Value);
  void_ptr IntVoidPtr = IntTypedPtr;
  int_ptr IntRoundTrip = static_cast<int_ptr>(IntVoidPtr);
  EXPECT_EQ(*IntRoundTrip, 123);
}

template <decorated IsDecorated> void checkConstVoidPointerConstConversions() {
  using const_void_ptr =
      multi_ptr_t<const void, address_space::private_space, IsDecorated>;
  using const_int_ptr =
      multi_ptr_t<const int, address_space::private_space, IsDecorated>;

  int Value = 456;
  auto MutablePtr =
      makePtr<int, address_space::private_space, IsDecorated>(&Value);

  // Convert mutable to const typed first, then to const void
  const_int_ptr ConstIntPtr1 = MutablePtr;
  const_void_ptr ConstVoidPtr = ConstIntPtr1;
  EXPECT_NE(ConstVoidPtr.get(), nullptr);

  const_int_ptr ConstIntPtr = static_cast<const_int_ptr>(ConstVoidPtr);
  EXPECT_EQ(*ConstIntPtr, 456);

  const int ConstValue = 789;
  auto ConstTypedPtr =
      makePtr<const int, address_space::private_space, IsDecorated>(
          &ConstValue);
  const_void_ptr ConstVoidPtr2 = ConstTypedPtr;
  const_int_ptr ConstIntPtr2 = static_cast<const_int_ptr>(ConstVoidPtr2);
  EXPECT_EQ(*ConstIntPtr2, 789);
}

template <decorated IsDecorated> void checkVoidPointerComparisons() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;

  TestData Data1{1, 1.0f};
  TestData Data2{2, 2.0f};

  auto TypedPtr1 =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data1);
  auto TypedPtr2 =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data2);

  void_ptr VoidPtr1 = TypedPtr1;
  void_ptr VoidPtr2 = TypedPtr2;
  void_ptr VoidPtr1Copy = TypedPtr1;

  EXPECT_TRUE(VoidPtr1 == VoidPtr1Copy);
  EXPECT_FALSE(VoidPtr1 != VoidPtr1Copy);
  EXPECT_TRUE(VoidPtr1 != VoidPtr2);
  EXPECT_FALSE(VoidPtr1 == VoidPtr2);

  void_ptr NullPtr = nullptr;
  EXPECT_TRUE(NullPtr == nullptr);
  EXPECT_FALSE(NullPtr != nullptr);
  EXPECT_TRUE(nullptr == NullPtr);
  EXPECT_FALSE(nullptr != NullPtr);

  EXPECT_FALSE(VoidPtr1 == nullptr);
  EXPECT_TRUE(VoidPtr1 != nullptr);
}

template <decorated IsDecorated> void checkVoidPointerGetMethods() {
  using void_ptr = multi_ptr_t<void, address_space::private_space, IsDecorated>;

  TestData Data{50, 5.0f};
  auto TypedPtr =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data);
  void_ptr VoidPtr = TypedPtr;

  EXPECT_NE(VoidPtr.get(), nullptr);

  if constexpr (IsDecorated == decorated::legacy) {
    EXPECT_EQ(VoidPtr.get_raw(), static_cast<void *>(&Data));
    EXPECT_EQ(VoidPtr.get(), VoidPtr.get_decorated());
  } else
    EXPECT_NE(VoidPtr.get(), nullptr);
}

template <decorated IsDecorated> void checkVoidPointerDecorationConversion() {
  using void_no =
      multi_ptr_t<void, address_space::private_space, decorated::no>;
  using void_yes =
      multi_ptr_t<void, address_space::private_space, decorated::yes>;

  TestData Data{100, 10.0f};
  auto TypedPtr =
      makePtr<TestData, address_space::private_space, IsDecorated>(&Data);

  if constexpr (IsDecorated != decorated::legacy) {
    auto VoidPtr = static_cast<
        multi_ptr_t<void, address_space::private_space, IsDecorated>>(TypedPtr);

    if constexpr (IsDecorated == decorated::no) {
      // Test implicit conversion to decorated::yes
      void_yes ToDecorated = VoidPtr;
      EXPECT_NE(ToDecorated.get(), nullptr);

      // Convert back to typed pointer to verify correctness
      auto TypedCheck = static_cast<
          multi_ptr_t<TestData, address_space::private_space, decorated::yes>>(
          ToDecorated);
      EXPECT_EQ(TypedCheck->Value, 100);
    } else {
      // Test implicit conversion to decorated::no
      void_no ToUndecorated = VoidPtr;
      EXPECT_NE(ToUndecorated.get(), nullptr);

      // Convert back to typed pointer to verify correctness
      auto TypedCheck = static_cast<
          multi_ptr_t<TestData, address_space::private_space, decorated::no>>(
          ToUndecorated);
      EXPECT_EQ(TypedCheck->Value, 100);
    }
  }
}

TEST(MultiPtrVoidSpecialization,
     DefaultAndNullptrConstructionCreatesNullPointers) {
  checkVoidPointerConstruction<decorated::no>();
  checkVoidPointerConstruction<decorated::yes>();
  checkVoidPointerConstruction<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, CopyAndMoveConstructionPreservesPointerValue) {
  checkVoidPointerCopyAndMove<decorated::no>();
  checkVoidPointerCopyAndMove<decorated::yes>();
  checkVoidPointerCopyAndMove<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, AssignmentOperatorsWorkCorrectly) {
  checkVoidPointerAssignment<decorated::no>();
  checkVoidPointerAssignment<decorated::yes>();
  checkVoidPointerAssignment<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, ExplicitConversionToTypedPointerWorks) {
  checkVoidPointerExplicitConversionToTyped<decorated::no>();
  checkVoidPointerExplicitConversionToTyped<decorated::yes>();
  checkVoidPointerExplicitConversionToTyped<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, ConstVoidPointerHandlesConstCorrectly) {
  checkConstVoidPointerConstConversions<decorated::no>();
  checkConstVoidPointerConstConversions<decorated::yes>();
  checkConstVoidPointerConstConversions<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, ComparisonOperatorsWorkCorrectly) {
  checkVoidPointerComparisons<decorated::no>();
  checkVoidPointerComparisons<decorated::yes>();
  checkVoidPointerComparisons<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, GetMethodsReturnCorrectPointers) {
  checkVoidPointerGetMethods<decorated::no>();
  checkVoidPointerGetMethods<decorated::yes>();
  checkVoidPointerGetMethods<decorated::legacy>();
}

TEST(MultiPtrVoidSpecialization, DecorationConversionsBetweenModesWork) {
  checkVoidPointerDecorationConversion<decorated::no>();
  checkVoidPointerDecorationConversion<decorated::yes>();
}

TEST(MultiPtrVoidSpecialization, VoidPointerWorksWithDifferentAddressSpaces) {
  using global_void =
      multi_ptr_t<void, address_space::global_space, decorated::no>;
  using local_void =
      multi_ptr_t<void, address_space::local_space, decorated::no>;
  using generic_void =
      multi_ptr_t<void, address_space::generic_space, decorated::no>;

  EXPECT_TRUE((std::is_default_constructible_v<global_void>));
  EXPECT_TRUE((std::is_default_constructible_v<local_void>));
  EXPECT_TRUE((std::is_default_constructible_v<generic_void>));

  global_void GlobalNull;
  local_void LocalNull;
  generic_void GenericNull;

  EXPECT_EQ(GlobalNull, nullptr);
  EXPECT_EQ(LocalNull, nullptr);
  EXPECT_EQ(GenericNull, nullptr);
}

TEST(MultiPtrVoidSpecialization, LegacyVoidPointerConstructsFromTypedPointer) {
  using void_legacy =
      multi_ptr_t<void, address_space::private_space, decorated::legacy>;
  using int_legacy =
      multi_ptr_t<int, address_space::private_space, decorated::legacy>;

  int Value = 999;
  int_legacy IntPtr{&Value};

  void_legacy VoidPtr{IntPtr};
  // Legacy void has get_raw()
  EXPECT_EQ(VoidPtr.get_raw(), static_cast<void *>(&Value));

  int_legacy RoundTrip = static_cast<int_legacy>(VoidPtr);
  EXPECT_EQ(RoundTrip.get_raw(), &Value);
  EXPECT_EQ(*RoundTrip, 999);
}

} // namespace
