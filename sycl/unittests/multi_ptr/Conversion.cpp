#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

struct Record {
	int Value;
	int Tag;
};

template <typename ElementType, address_space Space, decorated IsDecorated>
using multi_ptr_t = sycl::multi_ptr<ElementType, Space, IsDecorated>;

template <address_space Space>
using access_element_t =
		std::conditional_t<Space == address_space::constant_space, const Record,
											 Record>;

template <address_space Space>
using access_storage_t = std::remove_const_t<access_element_t<Space>>;

template <typename ElementType, address_space Space, decorated IsDecorated>
multi_ptr_t<ElementType, Space, IsDecorated>
makePtr(std::add_pointer_t<ElementType> Ptr) {
	if constexpr (IsDecorated == decorated::legacy) {
		return multi_ptr_t<ElementType, Space, IsDecorated>{Ptr};
	} else {
		return sycl::address_space_cast<Space, IsDecorated>(Ptr);
	}
}

template <decorated IsDecorated> constexpr decorated toggledDecoration() {
	static_assert(IsDecorated != decorated::legacy);
	return IsDecorated == decorated::no ? decorated::yes : decorated::no;
}

template <address_space Space, decorated IsDecorated>
void checkPointerAccessAndUnderlyingPointers() {
	using element_t = access_element_t<Space>;
	using ptr_t = multi_ptr_t<element_t, Space, IsDecorated>;

	access_storage_t<Space> Values[] = {{11, 111}, {22, 222}};
	auto *RawPtr = static_cast<std::add_pointer_t<element_t>>(&Values[0]);
	ptr_t Ptr = makePtr<element_t, Space, IsDecorated>(RawPtr);

	EXPECT_EQ(Ptr->Value, 11);
	EXPECT_EQ(Ptr->Tag, 111);
	EXPECT_EQ(Ptr[1].Value, 22);
	EXPECT_EQ(Ptr[1].Tag, 222);

	decltype(Ptr.get()) PointerValue = Ptr;
	EXPECT_EQ(PointerValue, Ptr.get());
	EXPECT_EQ(Ptr.get_raw(), RawPtr);

	if constexpr (IsDecorated == decorated::legacy) {
		EXPECT_EQ(Ptr.get_decorated(), Ptr.get());
	} else {
		auto DecoratedPtr = makePtr<element_t, Space, decorated::yes>(RawPtr);
		EXPECT_EQ(Ptr.get_decorated(), DecoratedPtr.get_decorated());
	}
}

template <decorated SourceDecoration, decorated TargetDecoration>
void checkGenericExplicitMutableConversions() {
	using source_t =
			multi_ptr_t<Record, address_space::generic_space, SourceDecoration>;
	using private_t =
			multi_ptr_t<Record, address_space::private_space, TargetDecoration>;
	using global_t =
			multi_ptr_t<Record, address_space::global_space, TargetDecoration>;
	using local_t =
			multi_ptr_t<Record, address_space::local_space, TargetDecoration>;
	using const_private_t =
			multi_ptr_t<const Record, address_space::private_space, TargetDecoration>;
	using const_global_t =
			multi_ptr_t<const Record, address_space::global_space, TargetDecoration>;
	using const_local_t =
			multi_ptr_t<const Record, address_space::local_space, TargetDecoration>;

	Record Value{31, 131};
	source_t Source =
			makePtr<Record, address_space::generic_space, SourceDecoration>(&Value);

	EXPECT_TRUE((std::is_constructible_v<private_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<global_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<local_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<const_private_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<const_global_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<const_local_t, source_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, private_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, global_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, local_t>));

	private_t Private{Source};
	global_t Global{Source};
	local_t Local{Source};
	const_private_t ConstPrivate{Source};
	const_global_t ConstGlobal{Source};
	const_local_t ConstLocal{Source};

	EXPECT_EQ(Private.get_raw(), &Value);
	EXPECT_EQ(Global.get_raw(), &Value);
	EXPECT_EQ(Local.get_raw(), &Value);
	EXPECT_EQ(ConstPrivate.get_raw(), &Value);
	EXPECT_EQ(ConstGlobal.get_raw(), &Value);
	EXPECT_EQ(ConstLocal.get_raw(), &Value);
}

template <decorated SourceDecoration, decorated TargetDecoration>
void checkGenericExplicitConstConversions() {
	using source_t =
			multi_ptr_t<const Record, address_space::generic_space, SourceDecoration>;
	using private_t =
			multi_ptr_t<const Record, address_space::private_space, TargetDecoration>;
	using global_t =
			multi_ptr_t<const Record, address_space::global_space, TargetDecoration>;
	using local_t =
			multi_ptr_t<const Record, address_space::local_space, TargetDecoration>;

	const Record Value{32, 132};
	source_t Source = makePtr<const Record, address_space::generic_space,
														SourceDecoration>(&Value);

	EXPECT_TRUE((std::is_constructible_v<private_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<global_t, source_t>));
	EXPECT_TRUE((std::is_constructible_v<local_t, source_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, private_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, global_t>));
	EXPECT_FALSE((std::is_convertible_v<source_t, local_t>));

	private_t Private{Source};
	global_t Global{Source};
	local_t Local{Source};

	EXPECT_EQ(Private.get_raw(), &Value);
	EXPECT_EQ(Global.get_raw(), &Value);
	EXPECT_EQ(Local.get_raw(), &Value);
}

template <address_space Space, decorated SourceDecoration>
void checkNonLegacyMutableSameSpaceConversions() {
	static_assert(SourceDecoration != decorated::legacy);

	using source_t = multi_ptr_t<Record, Space, SourceDecoration>;
	using typed_no_t = multi_ptr_t<Record, Space, decorated::no>;
	using typed_yes_t = multi_ptr_t<Record, Space, decorated::yes>;
	using typed_legacy_t = multi_ptr_t<Record, Space, decorated::legacy>;
	using toggled_t =
			multi_ptr_t<Record, Space, toggledDecoration<SourceDecoration>()>;
	using void_same_t = multi_ptr_t<void, Space, SourceDecoration>;
	using void_toggled_t =
			multi_ptr_t<void, Space, toggledDecoration<SourceDecoration>()>;
	using void_no_t = multi_ptr_t<void, Space, decorated::no>;
	using void_yes_t = multi_ptr_t<void, Space, decorated::yes>;
	using void_legacy_t = multi_ptr_t<void, Space, decorated::legacy>;
	using const_no_t = multi_ptr_t<const Record, Space, decorated::no>;
	using const_yes_t = multi_ptr_t<const Record, Space, decorated::yes>;
	using const_legacy_t = multi_ptr_t<const Record, Space, decorated::legacy>;

	Record Value{41, 141};
	source_t Source = makePtr<Record, Space, SourceDecoration>(&Value);

	EXPECT_TRUE((std::is_convertible_v<source_t, toggled_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, void_no_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, void_yes_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, void_legacy_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_no_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_yes_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_legacy_t>));
	EXPECT_TRUE((std::is_constructible_v<source_t, void_same_t>));
	EXPECT_FALSE((std::is_convertible_v<void_same_t, source_t>));

	toggled_t Toggled = Source;
	void_no_t VoidNo = Source;
	void_yes_t VoidYes = Source;
	void_legacy_t VoidLegacy = Source;
	const_no_t ConstNo = Source;
	const_yes_t ConstYes = Source;
	const_legacy_t ConstLegacy = Source;
	void_same_t VoidSame = Source;
	void_toggled_t VoidToggled = VoidSame;
	typed_no_t FromVoidNo = static_cast<typed_no_t>(VoidNo);
	typed_yes_t FromVoidYes = static_cast<typed_yes_t>(VoidYes);
	typed_legacy_t FromVoidLegacy = static_cast<typed_legacy_t>(VoidLegacy);
	toggled_t FromVoidToggled = static_cast<toggled_t>(VoidToggled);
	source_t FromVoid = static_cast<source_t>(VoidSame);

	EXPECT_EQ(Toggled.get_raw(), &Value);
	EXPECT_EQ(ConstNo.get_raw(), &Value);
	EXPECT_EQ(ConstYes.get_raw(), &Value);
	EXPECT_EQ(ConstLegacy.get_raw(), &Value);
	EXPECT_EQ(FromVoidNo.get_raw(), &Value);
	EXPECT_EQ(FromVoidYes.get_raw(), &Value);
	EXPECT_EQ(FromVoidLegacy.get_raw(), &Value);
	EXPECT_EQ(FromVoidToggled.get_raw(), &Value);
	EXPECT_EQ(FromVoid.get_raw(), &Value);
}

template <address_space Space, decorated SourceDecoration>
void checkNonLegacyConstSameSpaceConversions() {
	static_assert(SourceDecoration != decorated::legacy);

	using source_t = multi_ptr_t<const Record, Space, SourceDecoration>;
	using typed_no_t = multi_ptr_t<const Record, Space, decorated::no>;
	using typed_yes_t = multi_ptr_t<const Record, Space, decorated::yes>;
	using typed_legacy_t = multi_ptr_t<const Record, Space, decorated::legacy>;
	using toggled_t =
			multi_ptr_t<const Record, Space, toggledDecoration<SourceDecoration>()>;
	using const_void_same_t = multi_ptr_t<const void, Space, SourceDecoration>;
	using const_void_toggled_t =
			multi_ptr_t<const void, Space, toggledDecoration<SourceDecoration>()>;
	using const_void_no_t = multi_ptr_t<const void, Space, decorated::no>;
	using const_void_yes_t = multi_ptr_t<const void, Space, decorated::yes>;
	using const_void_legacy_t = multi_ptr_t<const void, Space, decorated::legacy>;

	const Record Value{42, 142};
	source_t Source = makePtr<const Record, Space, SourceDecoration>(&Value);

	EXPECT_TRUE((std::is_convertible_v<source_t, toggled_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_void_no_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_void_yes_t>));
	EXPECT_TRUE((std::is_convertible_v<source_t, const_void_legacy_t>));
	EXPECT_TRUE((std::is_constructible_v<source_t, const_void_same_t>));
	EXPECT_FALSE((std::is_convertible_v<const_void_same_t, source_t>));

	toggled_t Toggled = Source;
	const_void_no_t ConstVoidNo = Source;
	const_void_yes_t ConstVoidYes = Source;
	const_void_legacy_t ConstVoidLegacy = Source;
	const_void_same_t ConstVoidSame = Source;
	const_void_toggled_t ConstVoidToggled = ConstVoidSame;
	typed_no_t FromConstVoidNo = static_cast<typed_no_t>(ConstVoidNo);
	typed_yes_t FromConstVoidYes = static_cast<typed_yes_t>(ConstVoidYes);
	typed_legacy_t FromConstVoidLegacy = static_cast<typed_legacy_t>(ConstVoidLegacy);
	toggled_t FromConstVoidToggled = static_cast<toggled_t>(ConstVoidToggled);
	source_t FromConstVoid = static_cast<source_t>(ConstVoidSame);

	EXPECT_EQ(Toggled.get_raw(), &Value);
	EXPECT_EQ(FromConstVoidNo.get_raw(), &Value);
	EXPECT_EQ(FromConstVoidYes.get_raw(), &Value);
	EXPECT_EQ(FromConstVoidLegacy.get_raw(), &Value);
	EXPECT_EQ(FromConstVoidToggled.get_raw(), &Value);
	EXPECT_EQ(FromConstVoid.get_raw(), &Value);
}

template <address_space Space> void checkLegacyConversions() {
	using mutable_t = multi_ptr_t<Record, Space, decorated::legacy>;
	using const_t = multi_ptr_t<const Record, Space, decorated::legacy>;
	using void_t = multi_ptr_t<void, Space, decorated::legacy>;
	using const_void_t = multi_ptr_t<const void, Space, decorated::legacy>;

	Record MutableValue{51, 151};
	const Record ConstValue{52, 152};

	mutable_t Mutable = makePtr<Record, Space, decorated::legacy>(&MutableValue);
	const_t Const = makePtr<const Record, Space, decorated::legacy>(&ConstValue);

	EXPECT_TRUE((std::is_convertible_v<mutable_t, void_t>));
	EXPECT_TRUE((std::is_convertible_v<mutable_t, const_t>));
	EXPECT_TRUE((std::is_convertible_v<const_t, const_void_t>));
	EXPECT_TRUE((std::is_constructible_v<mutable_t, void_t>));
	EXPECT_TRUE((std::is_constructible_v<const_t, const_void_t>));

	void_t VoidPtr = Mutable;
	const_t ConstFromMutable = Mutable;
	const_void_t ConstVoidPtr = Const;
	const_void_t ConstVoidFromVoid = VoidPtr;
	mutable_t MutableRoundTrip = static_cast<mutable_t>(VoidPtr);
	const_t ConstRoundTrip = static_cast<const_t>(ConstVoidPtr);

	EXPECT_EQ(VoidPtr.get_raw(), static_cast<void *>(&MutableValue));
	EXPECT_EQ(ConstFromMutable.get_raw(), &MutableValue);
	EXPECT_EQ(ConstVoidPtr.get_raw(), static_cast<const void *>(&ConstValue));
	EXPECT_EQ(ConstVoidFromVoid.get_raw(), static_cast<const void *>(&MutableValue));
	EXPECT_EQ(MutableRoundTrip.get_raw(), &MutableValue);
	EXPECT_EQ(ConstRoundTrip.get_raw(), &ConstValue);
}

template <decorated IsDecorated> void checkFactoryFunctions() {
	using ptr_t = multi_ptr_t<int, address_space::private_space, IsDecorated>;

	int Value = 77;
	auto CastResult =
			sycl::address_space_cast<address_space::private_space, IsDecorated>(
					&Value);
	using cast_result_t = decltype(CastResult);
	EXPECT_TRUE((std::is_same_v<cast_result_t, ptr_t>));
	EXPECT_EQ(CastResult.get_raw(), &Value);

	auto NullCast =
			sycl::address_space_cast<address_space::private_space, IsDecorated>(
					static_cast<int *>(nullptr));
	EXPECT_EQ(NullCast, nullptr);

	auto MakeResult =
			sycl::make_ptr<int, address_space::private_space, IsDecorated>(
					CastResult.get());
	using make_result_t = decltype(MakeResult);
	EXPECT_TRUE((std::is_same_v<make_result_t, ptr_t>));
	EXPECT_EQ(MakeResult.get_raw(), &Value);

	using underlying_pointer_t = decltype(CastResult.get());
	underlying_pointer_t NullPointer = nullptr;
	auto NullMake =
			sycl::make_ptr<int, address_space::private_space, IsDecorated>(
					NullPointer);
	EXPECT_EQ(NullMake, nullptr);
}

void checkDefaultLegacyMakePtr() {
	int Value = 88;
	auto LegacyPtr =
			sycl::make_ptr<int, address_space::private_space>(&Value);
	EXPECT_TRUE((std::is_same_v<decltype(LegacyPtr),
									 multi_ptr_t<int, address_space::private_space,
												 decorated::legacy>>));
	EXPECT_EQ(LegacyPtr.get_raw(), &Value);
}

TEST(MultiPtrConversion,
		 PointerAccessPointerConversionAndGetDecoratedCoverAllEnumValues) {
	checkPointerAccessAndUnderlyingPointers<address_space::global_space,
																					decorated::no>();
	checkPointerAccessAndUnderlyingPointers<address_space::global_space,
																					decorated::yes>();
	checkPointerAccessAndUnderlyingPointers<address_space::global_space,
																					decorated::legacy>();
	checkPointerAccessAndUnderlyingPointers<address_space::local_space,
																					decorated::no>();
	checkPointerAccessAndUnderlyingPointers<address_space::local_space,
																					decorated::yes>();
	checkPointerAccessAndUnderlyingPointers<address_space::local_space,
																					decorated::legacy>();
	checkPointerAccessAndUnderlyingPointers<address_space::constant_space,
																					decorated::no>();
	checkPointerAccessAndUnderlyingPointers<address_space::constant_space,
																					decorated::yes>();
	checkPointerAccessAndUnderlyingPointers<address_space::constant_space,
																					decorated::legacy>();
	checkPointerAccessAndUnderlyingPointers<address_space::private_space,
																					decorated::no>();
	checkPointerAccessAndUnderlyingPointers<address_space::private_space,
																					decorated::yes>();
	checkPointerAccessAndUnderlyingPointers<address_space::private_space,
																					decorated::legacy>();
	checkPointerAccessAndUnderlyingPointers<address_space::generic_space,
																					decorated::no>();
	checkPointerAccessAndUnderlyingPointers<address_space::generic_space,
																					decorated::yes>();
	checkPointerAccessAndUnderlyingPointers<address_space::generic_space,
																					decorated::legacy>();
}

TEST(MultiPtrConversion,
		 GenericSpaceExplicitConversionsSupportAllNonLegacyDecorationPairs) {
	checkGenericExplicitMutableConversions<decorated::no, decorated::no>();
	checkGenericExplicitMutableConversions<decorated::no, decorated::yes>();
	checkGenericExplicitMutableConversions<decorated::yes, decorated::no>();
	checkGenericExplicitMutableConversions<decorated::yes, decorated::yes>();

	checkGenericExplicitConstConversions<decorated::no, decorated::no>();
	checkGenericExplicitConstConversions<decorated::no, decorated::yes>();
	checkGenericExplicitConstConversions<decorated::yes, decorated::no>();
	checkGenericExplicitConstConversions<decorated::yes, decorated::yes>();
}

TEST(MultiPtrConversion,
		 NonLegacySameSpaceConversionsCoverVoidConstAndDecorationChanges) {
	checkNonLegacyMutableSameSpaceConversions<address_space::global_space,
																						decorated::no>();
	checkNonLegacyMutableSameSpaceConversions<address_space::global_space,
																						decorated::yes>();
	checkNonLegacyMutableSameSpaceConversions<address_space::local_space,
																						decorated::no>();
	checkNonLegacyMutableSameSpaceConversions<address_space::local_space,
																						decorated::yes>();
	checkNonLegacyMutableSameSpaceConversions<address_space::private_space,
																						decorated::no>();
	checkNonLegacyMutableSameSpaceConversions<address_space::private_space,
																						decorated::yes>();
	checkNonLegacyMutableSameSpaceConversions<address_space::generic_space,
																						decorated::no>();
	checkNonLegacyMutableSameSpaceConversions<address_space::generic_space,
																						decorated::yes>();

	checkNonLegacyConstSameSpaceConversions<address_space::global_space,
																					decorated::no>();
	checkNonLegacyConstSameSpaceConversions<address_space::global_space,
																					decorated::yes>();
	checkNonLegacyConstSameSpaceConversions<address_space::local_space,
																					decorated::no>();
	checkNonLegacyConstSameSpaceConversions<address_space::local_space,
																					decorated::yes>();
	checkNonLegacyConstSameSpaceConversions<address_space::constant_space,
																					decorated::no>();
	checkNonLegacyConstSameSpaceConversions<address_space::constant_space,
																					decorated::yes>();
	checkNonLegacyConstSameSpaceConversions<address_space::private_space,
																					decorated::no>();
	checkNonLegacyConstSameSpaceConversions<address_space::private_space,
																					decorated::yes>();
	checkNonLegacyConstSameSpaceConversions<address_space::generic_space,
																					decorated::no>();
	checkNonLegacyConstSameSpaceConversions<address_space::generic_space,
																					decorated::yes>();
}

TEST(MultiPtrConversion, LegacyConversionsCoverAllAddressSpaces) {
	checkLegacyConversions<address_space::global_space>();
	checkLegacyConversions<address_space::local_space>();
	checkLegacyConversions<address_space::constant_space>();
	checkLegacyConversions<address_space::private_space>();
	checkLegacyConversions<address_space::generic_space>();
}

TEST(MultiPtrConversion, FactoryFunctionsPreservePrivatePointersAndNullptr) {
	checkFactoryFunctions<decorated::no>();
	checkFactoryFunctions<decorated::yes>();
	checkFactoryFunctions<decorated::legacy>();
	checkDefaultLegacyMakePtr();
}

} // namespace

