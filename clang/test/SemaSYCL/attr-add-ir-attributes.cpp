// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// Tests valid and invalid arguments in __sycl_detail__::add_ir_attributes_*
// attributes.

enum TestEnum {
  EnumVal1,
  EnumVal2
};

constexpr decltype(nullptr) CENullptr = nullptr;
constexpr const char CEStr[] = "Text";
constexpr int CEInt = 1;
constexpr float CEFloat = 3.14;
constexpr bool CETrue = true;
constexpr bool CEFalse = false;
constexpr TestEnum CEEnum = TestEnum::EnumVal1;
constexpr char CEChar = 'F';

constexpr const char CEAttrName1[] = "CEAttr1";
constexpr const char CEAttrName2[] = "CEAttr2";
constexpr const char CEAttrName3[] = "CEAttr3";
constexpr const char CEAttrName4[] = "CEAttr4";
constexpr const char CEAttrName5[] = "CEAttr5";
constexpr const char CEAttrName6[] = "CEAttr6";
constexpr const char CEAttrName7[] = "CEAttr7";
constexpr const char CEAttrName8[] = "CEAttr7";

[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr)]] void FunctionLiteral1() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Text")]] void FunctionLiteral2() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] void FunctionLiteral3() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 3.14)]] void FunctionLiteral4() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", true)]] void FunctionLiteral5() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", false)]] void FunctionLiteral6() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", TestEnum::EnumVal1)]] void FunctionLiteral7() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 'F')]] void FunctionLiteral8() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void FunctionLiteral9() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", nullptr)]] void FunctionLiteral10() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Text")]] void FunctionLiteral11() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", 1)]] void FunctionLiteral12() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", 3.14)]] void FunctionLiteral13() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", true)]] void FunctionLiteral14() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", false)]] void FunctionLiteral15() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] void FunctionLiteral16() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", 'F')]] void FunctionLiteral17() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void FunctionLiteral18() {}

[[__sycl_detail__::add_ir_attributes_function("Attr1", CENullptr)]] void FunctionCEVal1() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEStr)]] void FunctionCEVal2() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEInt)]] void FunctionCEVal3() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEFloat)]] void FunctionCEVal4() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CETrue)]] void FunctionCEVal5() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEFalse)]] void FunctionCEVal6() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEEnum)]] void FunctionCEVal7() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEChar)]] void FunctionCEVal8() {}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void FunctionCEVal9() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CENullptr)]] void FunctionCEVal10() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEStr)]] void FunctionCEVal11() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEInt)]] void FunctionCEVal12() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEFloat)]] void FunctionCEVal13() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CETrue)]] void FunctionCEVal14() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEFalse)]] void FunctionCEVal15() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEEnum)]] void FunctionCEVal16() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEChar)]] void FunctionCEVal17() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void FunctionCEVal18() {}

[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, nullptr)]] void FunctionCEName1() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, "Text")]] void FunctionCEName2() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, 1)]] void FunctionCEName3() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, 3.14)]] void FunctionCEName4() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, true)]] void FunctionCEName5() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, false)]] void FunctionCEName6() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, TestEnum::EnumVal1)]] void FunctionCEName7() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, 'F')]] void FunctionCEName87() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void FunctionCEName9() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] void FunctionCEName10() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, "Text")]] void FunctionCEName11() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, 1)]] void FunctionCEName12() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] void FunctionCEName13() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, true)]] void FunctionCEName14() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, false)]] void FunctionCEName15() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] void FunctionCEName16() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, 'F')]] void FunctionCEName17() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void FunctionCEName18() {}

[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CENullptr)]] void FunctionCE1() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEStr)]] void FunctionCE2() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEInt)]] void FunctionCE3() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEFloat)]] void FunctionCE4() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CETrue)]] void FunctionCE5() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEFalse)]] void FunctionCE6() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEEnum)]] void FunctionCE7() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEChar)]] void FunctionCE8() {}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void FunctionCE9() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] void FunctionCE10() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] void FunctionCE11() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] void FunctionCE12() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] void FunctionCE13() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] void FunctionCE14() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] void FunctionCE15() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] void FunctionCE16() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEChar)]] void FunctionCE17() {}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void FunctionCE18() {}

template <decltype(nullptr) Null> [[__sycl_detail__::add_ir_attributes_function("Attr1", Null)]] void FunctionTemplate1() {}
template <const char *Str> [[__sycl_detail__::add_ir_attributes_function("Attr1", Str)]] void FunctionTemplate2() {}
template <int I> [[__sycl_detail__::add_ir_attributes_function("Attr1", I)]] void FunctionTemplate3() {}
template <bool B> [[__sycl_detail__::add_ir_attributes_function("Attr1", B)]] void FunctionTemplate4() {}
template <TestEnum E> [[__sycl_detail__::add_ir_attributes_function("Attr1", E)]] void FunctionTemplate5() {}
template <char C> [[__sycl_detail__::add_ir_attributes_function("Attr1", C)]] void FunctionTemplate6() {}
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> [[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", Null, Str, I, B, E, C)]] void FunctionTemplate7() {}
template <decltype(nullptr) Null> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, Null)]] void FunctionTemplate8() {}
template <const char *Str> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, Str)]] void FunctionTemplate9() {}
template <int I> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, I)]] void FunctionTemplate10() {}
template <bool B> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, B)]] void FunctionTemplate11() {}
template <TestEnum E> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, E)]] void FunctionTemplate12() {}
template <char C> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, C)]] void FunctionTemplate13() {}
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> [[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, Null, Str, I, B, E, C)]] void FunctionTemplate14() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, nullptr)]] void FunctionTemplate15() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, "Text")]] void FunctionTemplate16() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, 1)]] void FunctionTemplate17() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, 3.14)]] void FunctionTemplate18() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, true)]] void FunctionTemplate19() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, false)]] void FunctionTemplate20() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, TestEnum::EnumVal1)]] void FunctionTemplate21() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, 'F')]] void FunctionTemplate22() {}
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> [[__sycl_detail__::add_ir_attributes_function(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void FunctionTemplate23() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CENullptr)]] void FunctionTemplate24() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEStr)]] void FunctionTemplate25() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEInt)]] void FunctionTemplate26() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEFloat)]] void FunctionTemplate27() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CETrue)]] void FunctionTemplate28() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEFalse)]] void FunctionTemplate29() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEEnum)]] void FunctionTemplate30() {}
template <const char *Name> [[__sycl_detail__::add_ir_attributes_function(Name, CEChar)]] void FunctionTemplate31() {}
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> [[__sycl_detail__::add_ir_attributes_function(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void FunctionTemplate32() {}
void InstantiateFunctionTemplates() {
  FunctionTemplate1<nullptr>();
  FunctionTemplate1<CENullptr>();
  FunctionTemplate2<CEStr>();
  FunctionTemplate3<1>();
  FunctionTemplate3<CEInt>();
  FunctionTemplate4<true>();
  FunctionTemplate4<CETrue>();
  FunctionTemplate4<false>();
  FunctionTemplate4<CEFalse>();
  FunctionTemplate5<TestEnum::EnumVal1>();
  FunctionTemplate5<CEEnum>();
  FunctionTemplate6<'F'>();
  FunctionTemplate6<CEChar>();
  FunctionTemplate7<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'>();
  FunctionTemplate7<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar>();
  FunctionTemplate8<nullptr>();
  FunctionTemplate8<CENullptr>();
  FunctionTemplate9<CEStr>();
  FunctionTemplate10<1>();
  FunctionTemplate10<CEInt>();
  FunctionTemplate11<true>();
  FunctionTemplate11<CETrue>();
  FunctionTemplate11<false>();
  FunctionTemplate11<CEFalse>();
  FunctionTemplate12<TestEnum::EnumVal1>();
  FunctionTemplate12<CEEnum>();
  FunctionTemplate13<'F'>();
  FunctionTemplate13<CEChar>();
  FunctionTemplate14<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'>();
  FunctionTemplate14<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar>();
  FunctionTemplate15<CEAttrName1>();
  FunctionTemplate16<CEAttrName1>();
  FunctionTemplate17<CEAttrName1>();
  FunctionTemplate18<CEAttrName1>();
  FunctionTemplate19<CEAttrName1>();
  FunctionTemplate20<CEAttrName1>();
  FunctionTemplate21<CEAttrName1>();
  FunctionTemplate22<CEAttrName1>();
  FunctionTemplate23<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8>();
  FunctionTemplate24<CEAttrName1>();
  FunctionTemplate25<CEAttrName1>();
  FunctionTemplate26<CEAttrName1>();
  FunctionTemplate27<CEAttrName1>();
  FunctionTemplate28<CEAttrName1>();
  FunctionTemplate29<CEAttrName1>();
  FunctionTemplate30<CEAttrName1>();
  FunctionTemplate31<CEAttrName1>();
  FunctionTemplate32<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8>();
}

[[__sycl_detail__::add_ir_attributes_function("Attr1")]] void InvalidFunction1() {}                                                                                                                 // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr, "Attr2")]] void InvalidFunction2() {}                                                                                               // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", nullptr)]] void InvalidFunction3() {}                                                                                               // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1")]] void InvalidFunction4() {}                                                                                             // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]] void InvalidFunction5() {}                                                                           // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]] void InvalidFunction6() {}                                                                           // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", {"Attr5", "Attr3"}, nullptr)]] void InvalidFunction7() {}                                                                                    // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr, {"Attr5", "Attr3"})]] void InvalidFunction8() {}                                                                                    // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]] void InvalidFunction9() {}                                                                   // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}
[[__sycl_detail__::add_ir_attributes_function({1}, "Attr1", nullptr)]] void InvalidFunction10() {}                                                                                                  // expected-error {{initializer list in the first argument of 'add_ir_attributes_function' must contain only string literals}}
[[__sycl_detail__::add_ir_attributes_function({true, "Attr3"}, "Attr1", nullptr)]] void InvalidFunction11() {}                                                                                      // expected-error {{initializer list in the first argument of 'add_ir_attributes_function' must contain only string literals}}
[[__sycl_detail__::add_ir_attributes_function({"Attr3", 'c'}, "Attr1", nullptr)]] void InvalidFunction12() {}                                                                                       // expected-error {{initializer list in the first argument of 'add_ir_attributes_function' must contain only string literals}}
[[__sycl_detail__::add_ir_attributes_function(nullptr, "Attr1")]] void InvalidFunction13() {}                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(1, "Attr1")]] void InvalidFunction14() {}                                                                                                             // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(3.14, "Attr1")]] void InvalidFunction15() {}                                                                                                          // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(true, "Attr1")]] void InvalidFunction16() {}                                                                                                          // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(false, "Attr1")]] void InvalidFunction17() {}                                                                                                         // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(TestEnum::EnumVal1, "Attr1")]] void InvalidFunction18() {}                                                                                            // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function('F', "Attr1")]] void InvalidFunction19() {}                                                                                                           // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F', nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] void InvalidFunction20() {}         // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 3.14, "Attr3", 1, 3.14, true)]] void InvalidFunction21() {}                                                                                  // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CENullptr, "Attr1")]] void InvalidFunction22() {}                                                                                                     // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CEInt, "Attr1")]] void InvalidFunction23() {}                                                                                                         // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CEFloat, "Attr1")]] void InvalidFunction24() {}                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CETrue, "Attr1")]] void InvalidFunction25() {}                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CEFalse, "Attr1")]] void InvalidFunction26() {}                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CEEnum, "Attr1")]] void InvalidFunction27() {}                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CEChar, "Attr1")]] void InvalidFunction28() {}                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function(CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] void InvalidFunction29() {} // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEFloat, "Attr3", CEInt, CEFloat, CETrue)]] void InvalidFunction30() {}                                                                      // expected-error {{each attribute name in 'add_ir_attributes_function' must be either a string literal or a 'const char *' which is usable in a constant expression}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", &CEInt)]] void InvalidFunction31() {}                                                                                                        // expected-error {{each attribute argument in 'add_ir_attributes_function' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", 1, &CEInt, CEInt)]] void InvalidFunction32() {}                                                                            // expected-error {{each attribute argument in 'add_ir_attributes_function' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}

struct [[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] InvalidFunctionSubjectStruct;                   // expected-error {{'add_ir_attributes_function' attribute only applies to functions}}
void InvalidFunctionSubjectFunctionParameter([[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] int x) {} // expected-error {{'add_ir_attributes_function' attribute only applies to functions}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] int InvalidFunctionSubjectVar;                         // expected-error {{'add_ir_attributes_function' attribute only applies to functions}}

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr)]] GlobalVarStructLiteral1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Text")]] GlobalVarStructLiteral2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] GlobalVarStructLiteral3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 3.14)]] GlobalVarStructLiteral4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", true)]] GlobalVarStructLiteral5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", false)]] GlobalVarStructLiteral6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", TestEnum::EnumVal1)]] GlobalVarStructLiteral7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 'F')]] GlobalVarStructLiteral8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] GlobalVarStructLiteral9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", nullptr)]] GlobalVarStructLiteral10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Text")]] GlobalVarStructLiteral11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", 1)]] GlobalVarStructLiteral12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", 3.14)]] GlobalVarStructLiteral13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", true)]] GlobalVarStructLiteral14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", false)]] GlobalVarStructLiteral15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] GlobalVarStructLiteral16{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", 'F')]] GlobalVarStructLiteral17{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] GlobalVarStructLiteral18{};

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CENullptr)]] GlobalVarStructCEVal1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEStr)]] GlobalVarStructCEVal2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEInt)]] GlobalVarStructCEVal3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEFloat)]] GlobalVarStructCEVal4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CETrue)]] GlobalVarStructCEVal5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEFalse)]] GlobalVarStructCEVal6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEEnum)]] GlobalVarStructCEVal7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEChar)]] GlobalVarStructCEVal8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] GlobalVarStructCEVal9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CENullptr)]] GlobalVarStructCEVal10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEStr)]] GlobalVarStructCEVal11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEInt)]] GlobalVarStructCEVal12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEFloat)]] GlobalVarStructCEVal13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CETrue)]] GlobalVarStructCEVal14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEFalse)]] GlobalVarStructCEVal15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEEnum)]] GlobalVarStructCEVal16{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEChar)]] GlobalVarStructCEVal17{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] GlobalVarStructCEVal18{};

struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, nullptr)]] GlobalVarStructCEName1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, "Text")]] GlobalVarStructCEName2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, 1)]] GlobalVarStructCEName3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, 3.14)]] GlobalVarStructCEName4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, true)]] GlobalVarStructCEName5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, false)]] GlobalVarStructCEName6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, TestEnum::EnumVal1)]] GlobalVarStructCEName7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, 'F')]] GlobalVarStructCEName8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] GlobalVarStructCEName9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] GlobalVarStructCEName10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, "Text")]] GlobalVarStructCEName11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, 1)]] GlobalVarStructCEName12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] GlobalVarStructCEName13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, true)]] GlobalVarStructCEName14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, false)]] GlobalVarStructCEName15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] GlobalVarStructCEName16{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, 'F')]] GlobalVarStructCEName17{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] GlobalVarStructCEName18{};

struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CENullptr)]] GlobalVarStructCE1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEStr)]] GlobalVarStructCE2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEInt)]] GlobalVarStructCE3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEFloat)]] GlobalVarStructCE4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CETrue)]] GlobalVarStructCE5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEFalse)]] GlobalVarStructCE6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEEnum)]] GlobalVarStructCE7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEChar)]] GlobalVarStructCE8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] GlobalVarStructCE9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] GlobalVarStructCE10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] GlobalVarStructCE11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] GlobalVarStructCE12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] GlobalVarStructCE13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] GlobalVarStructCE14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] GlobalVarStructCE15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] GlobalVarStructCE16{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEChar)]] GlobalVarStructCE17{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] GlobalVarStructCE18{};

template <decltype(nullptr) Null> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", Null)]] GlobalVarStructTemplate1{};
template <const char *Str> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", Str)]] GlobalVarStructTemplate2{};
template <int I> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", I)]] GlobalVarStructTemplate3{};
template <bool B> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", B)]] GlobalVarStructTemplate4{};
template <TestEnum E> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", E)]] GlobalVarStructTemplate5{};
template <char C> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", C)]] GlobalVarStructTemplate6{};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", Null, Str, I, B, E, C)]] GlobalVarStructTemplate7{};
template <decltype(nullptr) Null> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, Null)]] GlobalVarStructTemplate8{};
template <const char *Str> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, Str)]] GlobalVarStructTemplate9{};
template <int I> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, I)]] GlobalVarStructTemplate10{};
template <bool B> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, B)]] GlobalVarStructTemplate11{};
template <TestEnum E> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, E)]] GlobalVarStructTemplate12{};
template <char C> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, C)]] GlobalVarStructTemplate13{};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, Null, Str, I, B, E, C)]] GlobalVarStructTemplate14{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, nullptr)]] GlobalVarStructTemplate15{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, "Text")]] GlobalVarStructTemplate16{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, 1)]] GlobalVarStructTemplate17{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, 3.14)]] GlobalVarStructTemplate18{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, true)]] GlobalVarStructTemplate19{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, false)]] GlobalVarStructTemplate20{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, TestEnum::EnumVal1)]] GlobalVarStructTemplate21{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, 'F')]] GlobalVarStructTemplate22{};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] GlobalVarStructTemplate23{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CENullptr)]] GlobalVarStructTemplate24{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEStr)]] GlobalVarStructTemplate25{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEInt)]] GlobalVarStructTemplate26{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEFloat)]] GlobalVarStructTemplate27{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CETrue)]] GlobalVarStructTemplate28{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEFalse)]] GlobalVarStructTemplate29{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEEnum)]] GlobalVarStructTemplate30{};
template <const char *Name> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name, CEChar)]] GlobalVarStructTemplate31{};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> struct [[__sycl_detail__::add_ir_attributes_global_variable(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] GlobalVarStructTemplate32{};
GlobalVarStructTemplate1<nullptr> InstantiatedGV1;
GlobalVarStructTemplate1<CENullptr> InstantiatedGV2;
GlobalVarStructTemplate2<CEStr> InstantiatedGV3;
GlobalVarStructTemplate3<1> InstantiatedGV4;
GlobalVarStructTemplate3<CEInt> InstantiatedGV5;
GlobalVarStructTemplate4<true> InstantiatedGV6;
GlobalVarStructTemplate4<CETrue> InstantiatedGV7;
GlobalVarStructTemplate4<false> InstantiatedGV8;
GlobalVarStructTemplate4<CEFalse> InstantiatedGV9;
GlobalVarStructTemplate5<TestEnum::EnumVal1> InstantiatedGV10;
GlobalVarStructTemplate5<CEEnum> InstantiatedGV11;
GlobalVarStructTemplate6<'F'> InstantiatedGV12;
GlobalVarStructTemplate6<CEChar> InstantiatedGV13;
GlobalVarStructTemplate7<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'> InstantiatedGV14;
GlobalVarStructTemplate7<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar> InstantiatedGV15;
GlobalVarStructTemplate8<nullptr> InstantiatedGV16;
GlobalVarStructTemplate8<CENullptr> InstantiatedGV17;
GlobalVarStructTemplate9<CEStr> InstantiatedGV18;
GlobalVarStructTemplate10<1> InstantiatedGV19;
GlobalVarStructTemplate10<CEInt> InstantiatedGV20;
GlobalVarStructTemplate11<true> InstantiatedGV21;
GlobalVarStructTemplate11<CETrue> InstantiatedGV22;
GlobalVarStructTemplate11<false> InstantiatedGV23;
GlobalVarStructTemplate11<CEFalse> InstantiatedGV24;
GlobalVarStructTemplate12<TestEnum::EnumVal1> InstantiatedGV25;
GlobalVarStructTemplate12<CEEnum> InstantiatedGV26;
GlobalVarStructTemplate13<'F'> InstantiatedGV27;
GlobalVarStructTemplate13<CEChar> InstantiatedGV28;
GlobalVarStructTemplate14<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'> InstantiatedGV29;
GlobalVarStructTemplate14<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar> InstantiatedGV30;
GlobalVarStructTemplate15<CEAttrName1> InstantiatedGV31;
GlobalVarStructTemplate16<CEAttrName1> InstantiatedGV32;
GlobalVarStructTemplate17<CEAttrName1> InstantiatedGV33;
GlobalVarStructTemplate18<CEAttrName1> InstantiatedGV34;
GlobalVarStructTemplate19<CEAttrName1> InstantiatedGV35;
GlobalVarStructTemplate20<CEAttrName1> InstantiatedGV36;
GlobalVarStructTemplate21<CEAttrName1> InstantiatedGV37;
GlobalVarStructTemplate22<CEAttrName1> InstantiatedGV38;
GlobalVarStructTemplate23<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8> InstantiatedGV39;
GlobalVarStructTemplate24<CEAttrName1> InstantiatedGV40;
GlobalVarStructTemplate25<CEAttrName1> InstantiatedGV41;
GlobalVarStructTemplate26<CEAttrName1> InstantiatedGV42;
GlobalVarStructTemplate27<CEAttrName1> InstantiatedGV43;
GlobalVarStructTemplate28<CEAttrName1> InstantiatedGV44;
GlobalVarStructTemplate29<CEAttrName1> InstantiatedGV45;
GlobalVarStructTemplate30<CEAttrName1> InstantiatedGV46;
GlobalVarStructTemplate31<CEAttrName1> InstantiatedGV47;
GlobalVarStructTemplate32<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8> InstantiatedGV48;

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1")]] InvalidGlobalVarStruct1{};                                                                                                                 // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr, "Attr2")]] InvalidGlobalVarStruct2{};                                                                                               // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", nullptr)]] InvalidGlobalVarStruct3{};                                                                                               // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1")]] InvalidGlobalVarStruct4{};                                                                                             // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]] InvalidGlobalVarStruct5{};                                                                           // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]] InvalidGlobalVarStruct6{};                                                                           // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", {"Attr5", "Attr3"}, nullptr)]] InvalidGlobalVarStruct7{};                                                                                    // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr, {"Attr5", "Attr3"})]] InvalidGlobalVarStruct8{};                                                                                    // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]] InvalidGlobalVarStruct9{};                                                                   // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({1}, "Attr1", nullptr)]] InvalidGlobalVarStruct10{};                                                                                                  // expected-error {{initializer list in the first argument of 'add_ir_attributes_global_variable' must contain only string literals}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({true, "Attr3"}, "Attr1", nullptr)]] InvalidGlobalVarStruct11{};                                                                                      // expected-error {{initializer list in the first argument of 'add_ir_attributes_global_variable' must contain only string literals}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr3", 'c'}, "Attr1", nullptr)]] InvalidGlobalVarStruct12{};                                                                                       // expected-error {{initializer list in the first argument of 'add_ir_attributes_global_variable' must contain only string literals}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(nullptr, "Attr1")]] InvalidGlobalVarStruct13{};                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(1, "Attr1")]] InvalidGlobalVarStruct14{};                                                                                                             // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(3.14, "Attr1")]] InvalidGlobalVarStruct15{};                                                                                                          // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(true, "Attr1")]] InvalidGlobalVarStruct16{};                                                                                                          // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(false, "Attr1")]] InvalidGlobalVarStruct17{};                                                                                                         // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(TestEnum::EnumVal1, "Attr1")]] InvalidGlobalVarStruct18{};                                                                                            // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable('F', "Attr1")]] InvalidGlobalVarStruct19{};                                                                                                           // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F', nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] InvalidGlobalVarStruct20{};         // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 3.14, "Attr3", 1, 3.14, true)]] InvalidGlobalVarStruct21{};                                                                                  // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CENullptr, "Attr1")]] InvalidGlobalVarStruct22{};                                                                                                     // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEInt, "Attr1")]] InvalidGlobalVarStruct23{};                                                                                                         // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEFloat, "Attr1")]] InvalidGlobalVarStruct24{};                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CETrue, "Attr1")]] InvalidGlobalVarStruct25{};                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEFalse, "Attr1")]] InvalidGlobalVarStruct26{};                                                                                                       // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEEnum, "Attr1")]] InvalidGlobalVarStruct27{};                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEChar, "Attr1")]] InvalidGlobalVarStruct28{};                                                                                                        // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable(CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] InvalidGlobalVarStruct29{}; // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEFloat, "Attr3", CEInt, CEFloat, CETrue)]] InvalidGlobalVarStruct30{};                                                                      // expected-error {{each attribute name in 'add_ir_attributes_global_variable' must be either a string literal or a 'const char *' which is usable in a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", &CEInt)]] InvalidGlobalVarStruct31{};                                                                                                        // expected-error {{each attribute argument in 'add_ir_attributes_global_variable' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", 1, &CEInt, CEInt)]] InvalidGlobalVarStruct32{};                                                                            // expected-error {{each attribute argument in 'add_ir_attributes_global_variable' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}

[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] void InvalidGlobalVariableSubjectFunction() {}               // expected-error {{'add_ir_attributes_global_variable' attribute only applies to structs, unions, and classes}}
void InvalidGlobalVariableSubjectFunctionParameter([[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] int x) {} // expected-error {{'add_ir_attributes_global_variable' attribute only applies to structs, unions, and classes}}
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] int InvalidGlobalVariableSubjectVar;                         // expected-error {{'add_ir_attributes_global_variable' attribute only applies to structs, unions, and classes}}

struct SpecialClassStructLiteral1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {}
};

struct __attribute__((sycl_special_class)) SpecialClassStructCEVal1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {}
};

struct __attribute__((sycl_special_class)) SpecialClassStructCEName1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, 'F')]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {}
};

struct __attribute__((sycl_special_class)) SpecialClassStructCE1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEChar)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {}
};

template <decltype(nullptr) Null> struct SpecialClassStructTemplate1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", Null)]] int x) {}
};
template <const char *Str> struct SpecialClassStructTemplate2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", Str)]] int x) {}
};
template <int I> struct SpecialClassStructTemplate3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", I)]] int x) {}
};
template <bool B> struct SpecialClassStructTemplate4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", B)]] int x) {}
};
template <TestEnum E> struct SpecialClassStructTemplate5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", E)]] int x) {}
};
template <char C> struct SpecialClassStructTemplate6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", C)]] int x) {}
};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> struct SpecialClassStructTemplate7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", Null, Str, I, B, E, C)]] int x) {}
};
template <decltype(nullptr) Null> struct SpecialClassStructTemplate8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, Null)]] int x) {}
};
template <const char *Str> struct SpecialClassStructTemplate9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, Str)]] int x) {}
};
template <int I> struct SpecialClassStructTemplate10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, I)]] int x) {}
};
template <bool B> struct SpecialClassStructTemplate11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, B)]] int x) {}
};
template <TestEnum E> struct SpecialClassStructTemplate12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, E)]] int x) {}
};
template <char C> struct SpecialClassStructTemplate13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, C)]] int x) {}
};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C> struct SpecialClassStructTemplate14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, Null, Str, I, B, E, C)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, nullptr)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, "Text")]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, 1)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, 3.14)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate19 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, true)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate20 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, false)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate21 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, TestEnum::EnumVal1)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate22 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, 'F')]] int x) {}
};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> struct SpecialClassStructTemplate23 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate24 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CENullptr)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate25 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEStr)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate26 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEInt)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate27 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEFloat)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate28 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CETrue)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate29 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEFalse)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate30 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEEnum)]] int x) {}
};
template <const char *Name> struct SpecialClassStructTemplate31 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name, CEChar)]] int x) {}
};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8> struct SpecialClassStructTemplate32 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {}
};
void InstantiateSpecialClassStructTemplates() {
  SpecialClassStructTemplate1<nullptr> InstantiatedSCS1;
  SpecialClassStructTemplate1<CENullptr> InstantiatedSCS2;
  SpecialClassStructTemplate2<CEStr> InstantiatedSCS3;
  SpecialClassStructTemplate3<1> InstantiatedSCS4;
  SpecialClassStructTemplate3<CEInt> InstantiatedSCS5;
  SpecialClassStructTemplate4<true> InstantiatedSCS6;
  SpecialClassStructTemplate4<CETrue> InstantiatedSCS7;
  SpecialClassStructTemplate4<false> InstantiatedSCS8;
  SpecialClassStructTemplate4<CEFalse> InstantiatedSCS9;
  SpecialClassStructTemplate5<TestEnum::EnumVal1> InstantiatedSCS10;
  SpecialClassStructTemplate5<CEEnum> InstantiatedSCS11;
  SpecialClassStructTemplate6<'F'> InstantiatedSCS12;
  SpecialClassStructTemplate6<CEChar> InstantiatedSCS13;
  SpecialClassStructTemplate7<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'> InstantiatedSCS14;
  SpecialClassStructTemplate7<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar> InstantiatedSCS15;
  SpecialClassStructTemplate8<nullptr> InstantiatedSCS16;
  SpecialClassStructTemplate8<CENullptr> InstantiatedSCS17;
  SpecialClassStructTemplate9<CEStr> InstantiatedSCS18;
  SpecialClassStructTemplate10<1> InstantiatedSCS19;
  SpecialClassStructTemplate10<CEInt> InstantiatedSCS20;
  SpecialClassStructTemplate11<true> InstantiatedSCS21;
  SpecialClassStructTemplate11<CETrue> InstantiatedSCS22;
  SpecialClassStructTemplate11<false> InstantiatedSCS23;
  SpecialClassStructTemplate11<CEFalse> InstantiatedSCS24;
  SpecialClassStructTemplate12<TestEnum::EnumVal1> InstantiatedSCS25;
  SpecialClassStructTemplate12<CEEnum> InstantiatedSCS26;
  SpecialClassStructTemplate13<'F'> InstantiatedSCS27;
  SpecialClassStructTemplate13<CEChar> InstantiatedSCS28;
  SpecialClassStructTemplate14<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F'> InstantiatedSCS29;
  SpecialClassStructTemplate14<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar> InstantiatedSCS30;
  SpecialClassStructTemplate15<CEAttrName1> InstantiatedSCS31;
  SpecialClassStructTemplate16<CEAttrName1> InstantiatedSCS32;
  SpecialClassStructTemplate17<CEAttrName1> InstantiatedSCS33;
  SpecialClassStructTemplate18<CEAttrName1> InstantiatedSCS34;
  SpecialClassStructTemplate19<CEAttrName1> InstantiatedSCS35;
  SpecialClassStructTemplate20<CEAttrName1> InstantiatedSCS36;
  SpecialClassStructTemplate21<CEAttrName1> InstantiatedSCS37;
  SpecialClassStructTemplate22<CEAttrName1> InstantiatedSCS38;
  SpecialClassStructTemplate23<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8> InstantiatedSCS39;
  SpecialClassStructTemplate24<CEAttrName1> InstantiatedSCS40;
  SpecialClassStructTemplate25<CEAttrName1> InstantiatedSCS41;
  SpecialClassStructTemplate26<CEAttrName1> InstantiatedSCS42;
  SpecialClassStructTemplate27<CEAttrName1> InstantiatedSCS43;
  SpecialClassStructTemplate28<CEAttrName1> InstantiatedSCS44;
  SpecialClassStructTemplate29<CEAttrName1> InstantiatedSCS45;
  SpecialClassStructTemplate30<CEAttrName1> InstantiatedSCS46;
  SpecialClassStructTemplate31<CEAttrName1> InstantiatedSCS47;
  SpecialClassStructTemplate32<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8> InstantiatedSCS48;
}

struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1")]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", nullptr, "Attr2")]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", nullptr)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr5", "Attr3"}, "Attr1")]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must have an attribute value for each attribute name}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct7 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", {"Attr5", "Attr3"}, nullptr)]] int x) {} // expected-error {{only the first argument of attribute 'add_ir_attributes_kernel_parameter' can be an initializer list}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct8 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", nullptr, {"Attr5", "Attr3"})]] int x) {} // expected-error {{only the first argument of attribute 'add_ir_attributes_kernel_parameter' can be an initializer list}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]] int x) {} // expected-error {{only the first argument of attribute 'add_ir_attributes_kernel_parameter' can be an initializer list}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({1}, "Attr1", nullptr)]] int x) {} // expected-error {{initializer list in the first argument of 'add_ir_attributes_kernel_parameter' must contain only string literals}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({true, "Attr3"}, "Attr1", nullptr)]] int x) {} // expected-error {{initializer list in the first argument of 'add_ir_attributes_kernel_parameter' must contain only string literals}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr3", 'c'}, "Attr1", nullptr)]] int x) {} // expected-error {{initializer list in the first argument of 'add_ir_attributes_kernel_parameter' must contain only string literals}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(nullptr, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(1, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(3.14, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(true, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct17 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(false, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct18 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(TestEnum::EnumVal1, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct19 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter('F', "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct20 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F', nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F')]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct21 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 3.14, "Attr3", 1, 3.14, true)]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct22 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CENullptr, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct23 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEInt, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct24 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEFloat, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct25 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CETrue, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct26 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEFalse, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct27 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEEnum, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct28 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEChar, "Attr1")]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct29 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar)]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct30 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", CEFloat, "Attr3", CEInt, CEFloat, CETrue)]] int x) {} // expected-error {{each attribute name in 'add_ir_attributes_kernel_parameter' must be either a string literal or a 'const char *' which is usable in a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct31 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", &CEInt)]] int x) {} // expected-error {{each attribute argument in 'add_ir_attributes_kernel_parameter' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
};
struct __attribute__((sycl_special_class)) InvalidSpecialClassStruct32 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", 1, &CEInt, CEInt)]] int x) {} // expected-error {{each attribute argument in 'add_ir_attributes_kernel_parameter' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
};

struct [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 1)]] InvalidKernelParameterSubjectStruct;     // expected-error {{'add_ir_attributes_kernel_parameter' attribute only applies to parameters}}
[[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 1)]] void InvalidKernelParameterSubjectFunction() {} // expected-error {{'add_ir_attributes_kernel_parameter' attribute only applies to parameters}}
[[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", 1)]] int InvalidKernelParameterSubjectVar;           // expected-error {{'add_ir_attributes_kernel_parameter' attribute only applies to parameters}}
