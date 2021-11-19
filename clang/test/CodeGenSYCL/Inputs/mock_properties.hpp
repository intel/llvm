#pragma once

enum TestEnum {
  Enum1 = 1,
  Enum2 = 2
};

template <const char *Name, const char *Value> struct StringProperty {
  static constexpr const char *name = Name;
  static constexpr const char *value = Value;
};

template <const char *Name, int Value> struct IntProperty {
  static constexpr const char *name = Name;
  static constexpr int value = Value;
};

template <const char *Name, bool Value> struct BoolProperty {
  static constexpr const char *name = Name;
  static constexpr bool value = Value;
};

template <const char *Name, TestEnum Value> struct TestEnumProperty {
  static constexpr const char *name = Name;
  static constexpr TestEnum value = Value;
};

template <const char *Name, decltype(nullptr) Value> struct NullptrProperty {
  static constexpr const char *name = Name;
  static constexpr decltype(nullptr) value = Value;
};

const char PropertyName1[] = "Prop1";
const char PropertyValue1[] = "Property string";
const char PropertyName2[] = "Prop2";
constexpr int PropertyValue2 = 1;
const char PropertyName3[] = "Prop3";
constexpr bool PropertyValue3 = true;
const char PropertyName4[] = "Prop4";
constexpr TestEnum PropertyValue4 = TestEnum::Enum2;
const char PropertyName5[] = "Prop5";
constexpr decltype(nullptr) PropertyValue5 = nullptr;
const char PropertyName6[] = "Prop6";
constexpr decltype(nullptr) PropertyValue6 = nullptr;

using prop1 = StringProperty<PropertyName1, PropertyValue1>;
using prop2 = IntProperty<PropertyName2, PropertyValue2>;
using prop3 = BoolProperty<PropertyName3, PropertyValue3>;
using prop4 = TestEnumProperty<PropertyName4, PropertyValue4>;
using prop5 = StringProperty<PropertyName5, PropertyValue5>;
using prop6 = NullptrProperty<PropertyName6, PropertyValue6>;
