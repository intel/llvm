//===---------------------------- Operators.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

template <typename ElementType, decorated IsDecorated>
using private_multi_ptr =
		sycl::multi_ptr<ElementType, address_space::private_space, IsDecorated>;

template <typename ElementType, decorated IsDecorated>
private_multi_ptr<ElementType, IsDecorated>
makePrivatePtr(std::add_pointer_t<ElementType> Ptr) {
	return sycl::address_space_cast<address_space::private_space, IsDecorated>(
			Ptr);
}

struct Record {
	int Value;
	int Tag;
};

template <decorated IsDecorated> void checkOperatorTraits() {
	using ptr_t = private_multi_ptr<Record, IsDecorated>;
	using diff_t = typename ptr_t::difference_type;

	EXPECT_TRUE(
			(std::is_same_v<decltype(*std::declval<const ptr_t &>()),
											typename ptr_t::reference>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>()[diff_t{0}]),
															typename ptr_t::reference>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>().operator->()),
															typename ptr_t::pointer>));

	EXPECT_TRUE((std::is_same_v<decltype(++std::declval<ptr_t &>()), ptr_t &>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<ptr_t &>()++), ptr_t>));
	EXPECT_TRUE((std::is_same_v<decltype(--std::declval<ptr_t &>()), ptr_t &>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<ptr_t &>()--), ptr_t>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<ptr_t &>() += diff_t{0}),
															ptr_t &>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<ptr_t &>() -= diff_t{0}),
															ptr_t &>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() + diff_t{0}),
															ptr_t>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() - diff_t{0}),
															ptr_t>));

	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() ==
																			 std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() !=
																			 std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() <
																			 std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() >
																			 std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() <=
																			 std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() >=
																			 std::declval<const ptr_t &>()),
															bool>));

	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() == nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() != nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() < nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() > nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() <= nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(std::declval<const ptr_t &>() >= nullptr),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr == std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr != std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr < std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr > std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr <= std::declval<const ptr_t &>()),
															bool>));
	EXPECT_TRUE((std::is_same_v<decltype(nullptr >= std::declval<const ptr_t &>()),
															bool>));
}

template <decorated IsDecorated> void checkAccessOperators() {
	Record Values[] = {{10, 100}, {20, 200}, {30, 300}};
	auto Ptr = makePrivatePtr<Record, IsDecorated>(&Values[0]);

	EXPECT_EQ((*Ptr).Value, 10);
	EXPECT_EQ((*Ptr).Tag, 100);
	EXPECT_EQ(Ptr->Value, 10);
	EXPECT_EQ(Ptr->Tag, 100);
	EXPECT_EQ(Ptr[1].Value, 20);
	EXPECT_EQ(Ptr[1].Tag, 200);
	EXPECT_EQ(Ptr[2].Value, 30);
	EXPECT_EQ(Ptr[2].Tag, 300);
}

template <decorated IsDecorated> void checkArithmeticOperators() {
	Record Values[] = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
	using ptr_t = private_multi_ptr<Record, IsDecorated>;
	using diff_t = typename ptr_t::difference_type;

	auto Ptr = makePrivatePtr<Record, IsDecorated>(&Values[0]);
	auto Expected0 = makePrivatePtr<Record, IsDecorated>(&Values[0]);
	auto Expected1 = makePrivatePtr<Record, IsDecorated>(&Values[1]);
	auto Expected2 = makePrivatePtr<Record, IsDecorated>(&Values[2]);
	auto Expected3 = makePrivatePtr<Record, IsDecorated>(&Values[3]);

	ptr_t &PrefixIncrement = ++Ptr;
	EXPECT_EQ(std::addressof(PrefixIncrement), std::addressof(Ptr));
	EXPECT_EQ(Ptr, Expected1);

	ptr_t PostfixIncrement = Ptr++;
	EXPECT_EQ(PostfixIncrement, Expected1);
	EXPECT_EQ(Ptr, Expected2);

	ptr_t &PrefixDecrement = --Ptr;
	EXPECT_EQ(std::addressof(PrefixDecrement), std::addressof(Ptr));
	EXPECT_EQ(Ptr, Expected1);

	ptr_t PostfixDecrement = Ptr--;
	EXPECT_EQ(PostfixDecrement, Expected1);
	EXPECT_EQ(Ptr, Expected0);

	ptr_t &PlusAssign = (Ptr += diff_t{3});
	EXPECT_EQ(std::addressof(PlusAssign), std::addressof(Ptr));
	EXPECT_EQ(Ptr, Expected3);

	ptr_t &MinusAssign = (Ptr -= diff_t{2});
	EXPECT_EQ(std::addressof(MinusAssign), std::addressof(Ptr));
	EXPECT_EQ(Ptr, Expected1);

	ptr_t PlusResult = Ptr + diff_t{2};
	EXPECT_EQ(PlusResult, Expected3);
	EXPECT_EQ(Ptr, Expected1);

	ptr_t MinusResult = Ptr - diff_t{1};
	EXPECT_EQ(MinusResult, Expected0);
	EXPECT_EQ(Ptr, Expected1);
}

template <decorated IsDecorated> void checkRelationalOperators() {
	int Values[] = {5, 6, 7};

	auto First = makePrivatePtr<int, IsDecorated>(&Values[0]);
	auto SameAsFirst = makePrivatePtr<int, IsDecorated>(&Values[0]);
	auto Second = makePrivatePtr<int, IsDecorated>(&Values[1]);

	EXPECT_TRUE(First == SameAsFirst);
	EXPECT_FALSE(First != SameAsFirst);
	EXPECT_FALSE(First < SameAsFirst);
	EXPECT_FALSE(First > SameAsFirst);
	EXPECT_TRUE(First <= SameAsFirst);
	EXPECT_TRUE(First >= SameAsFirst);

	EXPECT_FALSE(First == Second);
	EXPECT_TRUE(First != Second);
	EXPECT_TRUE(First < Second);
	EXPECT_FALSE(First > Second);
	EXPECT_TRUE(First <= Second);
	EXPECT_FALSE(First >= Second);

	EXPECT_FALSE(Second == First);
	EXPECT_TRUE(Second != First);
	EXPECT_FALSE(Second < First);
	EXPECT_TRUE(Second > First);
	EXPECT_FALSE(Second <= First);
	EXPECT_TRUE(Second >= First);
}

template <decorated IsDecorated> void checkNullptrComparisonOperators() {
	int Value = 9;
	auto NullPtr = private_multi_ptr<int, IsDecorated>{nullptr};
	auto NonNullPtr = makePrivatePtr<int, IsDecorated>(&Value);

	EXPECT_TRUE(NullPtr == nullptr);
	EXPECT_FALSE(NullPtr != nullptr);
	EXPECT_FALSE(NullPtr < nullptr);
	EXPECT_FALSE(NullPtr > nullptr);
	EXPECT_TRUE(NullPtr <= nullptr);
	EXPECT_TRUE(NullPtr >= nullptr);

	EXPECT_TRUE(nullptr == NullPtr);
	EXPECT_FALSE(nullptr != NullPtr);
	EXPECT_FALSE(nullptr < NullPtr);
	EXPECT_FALSE(nullptr > NullPtr);
	EXPECT_TRUE(nullptr <= NullPtr);
	EXPECT_TRUE(nullptr >= NullPtr);

	EXPECT_FALSE(NonNullPtr == nullptr);
	EXPECT_TRUE(NonNullPtr != nullptr);
	EXPECT_FALSE(nullptr == NonNullPtr);
	EXPECT_TRUE(nullptr != NonNullPtr);
}

TEST(MultiPtrOperators, OperatorTraitsAreSatisfied) {
	checkOperatorTraits<decorated::no>();
	checkOperatorTraits<decorated::yes>();
}

TEST(MultiPtrOperators, AccessOperatorsProvidePointerLikeSemantics) {
	checkAccessOperators<decorated::no>();
	checkAccessOperators<decorated::yes>();
}

TEST(MultiPtrOperators, ArithmeticOperatorsProvideRandomAccessTraversal) {
	checkArithmeticOperators<decorated::no>();
	checkArithmeticOperators<decorated::yes>();
}

TEST(MultiPtrOperators, RelationalOperatorsCompareUnderlyingLocations) {
	checkRelationalOperators<decorated::no>();
	checkRelationalOperators<decorated::yes>();
}

TEST(MultiPtrOperators, NullptrComparisonOperatorsHandleNullValuesBothWays) {
	checkNullptrComparisonOperators<decorated::no>();
	checkNullptrComparisonOperators<decorated::yes>();
}

} // namespace


