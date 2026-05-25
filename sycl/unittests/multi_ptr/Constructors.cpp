//===---------------------------- Constructors.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

template <typename ElementType, decorated IsDecorated>
using private_multi_ptr =
    sycl::multi_ptr<ElementType, address_space::private_space, IsDecorated>;

template <typename ElementType, decorated IsDecorated>
using local_multi_ptr =
    sycl::multi_ptr<ElementType, address_space::local_space, IsDecorated>;

template <typename ElementType, decorated IsDecorated>
private_multi_ptr<ElementType, IsDecorated>
makePrivatePtr(std::add_pointer_t<ElementType> Ptr) {
  return sycl::address_space_cast<address_space::private_space, IsDecorated>(
      Ptr);
}

template <decorated IsDecorated> void checkCopyConstruction() {
  int Value = 17;
  auto Original = makePrivatePtr<int, IsDecorated>(&Value);
  private_multi_ptr<int, IsDecorated> Copy{Original};

  EXPECT_EQ(Copy, Original);
  EXPECT_NE(Copy, nullptr);
}

template <decorated IsDecorated> void checkMoveConstruction() {
  int Value = 23;
  auto Source = makePrivatePtr<int, IsDecorated>(&Value);
  auto Expected = makePrivatePtr<int, IsDecorated>(&Value);
  private_multi_ptr<int, IsDecorated> Moved{std::move(Source)};

  EXPECT_EQ(Moved, Expected);
  EXPECT_NE(Moved, nullptr);
}

TEST(MultiPtrConstructors, ValidConstructorTraitsAreSatisfied) {
  EXPECT_TRUE(
      (std::is_default_constructible_v<private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE((
      std::is_default_constructible_v<private_multi_ptr<int, decorated::yes>>));
  EXPECT_TRUE((std::is_constructible_v<private_multi_ptr<int, decorated::no>,
                                       std::nullptr_t>));
  EXPECT_TRUE((std::is_constructible_v<private_multi_ptr<int, decorated::yes>,
                                       std::nullptr_t>));
  EXPECT_TRUE(
      (std::is_copy_constructible_v<private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE(
      (std::is_copy_constructible_v<private_multi_ptr<int, decorated::yes>>));
  EXPECT_TRUE(
      (std::is_move_constructible_v<private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE(
      (std::is_move_constructible_v<private_multi_ptr<int, decorated::yes>>));
  EXPECT_TRUE(
      (std::is_constructible_v<private_multi_ptr<const int, decorated::no>,
                               private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE(
      (std::is_constructible_v<private_multi_ptr<const int, decorated::yes>,
                               private_multi_ptr<int, decorated::yes>>));
  EXPECT_TRUE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::no>,
                               private_multi_ptr<int, decorated::yes>>));
  EXPECT_TRUE((std::is_constructible_v<private_multi_ptr<int, decorated::yes>,
                                       private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE(
      (std::is_constructible_v<private_multi_ptr<const int, decorated::yes>,
                               private_multi_ptr<int, decorated::no>>));
  EXPECT_TRUE(
      (std::is_constructible_v<private_multi_ptr<const int, decorated::no>,
                               private_multi_ptr<int, decorated::yes>>));
}

TEST(MultiPtrConstructors, InvalidConstructorTraitsAreRejected) {
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::no>,
                               private_multi_ptr<const int, decorated::no>>));
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::yes>,
                               private_multi_ptr<const int, decorated::yes>>));
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::no>,
                               private_multi_ptr<float, decorated::no>>));
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::yes>,
                               private_multi_ptr<float, decorated::yes>>));
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::no>,
                               local_multi_ptr<float, decorated::no>>));
  EXPECT_FALSE(
      (std::is_constructible_v<private_multi_ptr<int, decorated::yes>,
                               local_multi_ptr<float, decorated::yes>>));
}

TEST(MultiPtrConstructors, DefaultConstructedPointersAreNull) {
  private_multi_ptr<int, decorated::no> Undecorated;
  private_multi_ptr<int, decorated::yes> Decorated;

  EXPECT_EQ(Undecorated, nullptr);
  EXPECT_EQ(Decorated, nullptr);
}

TEST(MultiPtrConstructors, NullptrConstructedPointersAreNull) {
  private_multi_ptr<int, decorated::no> Undecorated{nullptr};
  private_multi_ptr<int, decorated::yes> Decorated{nullptr};

  EXPECT_EQ(Undecorated, nullptr);
  EXPECT_EQ(Decorated, nullptr);
}

TEST(MultiPtrConstructors, CopyConstructionPreservesPointerValue) {
  checkCopyConstruction<decorated::no>();
  checkCopyConstruction<decorated::yes>();
}

TEST(MultiPtrConstructors, MoveConstructionPreservesPointerValue) {
  checkMoveConstruction<decorated::no>();
  checkMoveConstruction<decorated::yes>();
}

TEST(MultiPtrConstructors, ConvertingConstructionAddsConstQualification) {
  int Value = 31;

  private_multi_ptr<int, decorated::no> MutableUndecorated =
      makePrivatePtr<int, decorated::no>(&Value);
  private_multi_ptr<int, decorated::yes> MutableDecorated =
      makePrivatePtr<int, decorated::yes>(&Value);

  private_multi_ptr<const int, decorated::no> ConstUndecorated{
      MutableUndecorated};
  private_multi_ptr<const int, decorated::yes> ConstDecorated{MutableDecorated};
  auto ExpectedUndecorated = makePrivatePtr<const int, decorated::no>(&Value);
  auto ExpectedDecorated = makePrivatePtr<const int, decorated::yes>(&Value);

  EXPECT_EQ(ConstUndecorated, ExpectedUndecorated);
  EXPECT_EQ(ConstDecorated, ExpectedDecorated);
}

TEST(MultiPtrConstructors,
     ConvertingConstructionBetweenDecorationModesPreservesPointerValue) {
  int Value = 47;

  private_multi_ptr<int, decorated::yes> DecoratedPtr =
      makePrivatePtr<int, decorated::yes>(&Value);
  private_multi_ptr<int, decorated::no> UndecoratedPtr =
      makePrivatePtr<int, decorated::no>(&Value);

  private_multi_ptr<int, decorated::no> FromDecorated{DecoratedPtr};
  private_multi_ptr<int, decorated::yes> FromUndecorated{UndecoratedPtr};
  auto ExpectedUndecorated = makePrivatePtr<int, decorated::no>(&Value);
  auto ExpectedDecorated = makePrivatePtr<int, decorated::yes>(&Value);

  EXPECT_EQ(FromDecorated, ExpectedUndecorated);
  EXPECT_EQ(FromUndecorated, ExpectedDecorated);
}

TEST(MultiPtrConstructors,
     ConvertingConstructionCanAddConstAndChangeDecorationTogether) {
  int Value = 59;

  private_multi_ptr<int, decorated::no> MutableUndecorated =
      makePrivatePtr<int, decorated::no>(&Value);
  private_multi_ptr<int, decorated::yes> MutableDecorated =
      makePrivatePtr<int, decorated::yes>(&Value);

  private_multi_ptr<const int, decorated::yes> ConstDecorated{
      MutableUndecorated};
  private_multi_ptr<const int, decorated::no> ConstUndecorated{
      MutableDecorated};
  auto ExpectedDecorated = makePrivatePtr<const int, decorated::yes>(&Value);
  auto ExpectedUndecorated = makePrivatePtr<const int, decorated::no>(&Value);

  EXPECT_EQ(ConstDecorated, ExpectedDecorated);
  EXPECT_EQ(ConstUndecorated, ExpectedUndecorated);
}

} // namespace
