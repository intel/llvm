// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// Tests valid and invalid arguments in __sycl_detail__::add_ir_attributes_*
// attributes.

enum TestEnum {
  EnumVal1,
  EnumVal2
};

constexpr decltype(nullptr) CENullptr = nullptr;
constexpr const char *CEStr = "Text";
constexpr int CEInt = 1;
constexpr float CEFloat = 3.14;
constexpr bool CETrue = true;
constexpr bool CEFalse = false;
constexpr TestEnum CEEnum = TestEnum::EnumVal1;

constexpr const char *CEAttrName1 = "CEAttr1";
constexpr const char *CEAttrName2 = "CEAttr2";
constexpr const char *CEAttrName3 = "CEAttr3";
constexpr const char *CEAttrName4 = "CEAttr4";
constexpr const char *CEAttrName5 = "CEAttr5";
constexpr const char *CEAttrName6 = "CEAttr6";
constexpr const char *CEAttrName7 = "CEAttr7";

[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr)]] void FunctionLiteral1(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Text")]] void FunctionLiteral2(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 1)]] void FunctionLiteral3(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", 3.14)]] void FunctionLiteral4(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", true)]] void FunctionLiteral5(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", false)]] void FunctionLiteral6(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", TestEnum::EnumVal1)]] void FunctionLiteral7(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] void FunctionLiteral8(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", nullptr)]] void FunctionLiteral9(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Text")]] void FunctionLiteral10(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", 1)]] void FunctionLiteral11(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", 3.14)]] void FunctionLiteral12(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", true)]] void FunctionLiteral13(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", false)]] void FunctionLiteral14(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] void FunctionLiteral15(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] void FunctionLiteral16(){}

[[__sycl_detail__::add_ir_attributes_function("Attr1", CENullptr)]] void FunctionCEVal1(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEStr)]] void FunctionCEVal2(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEInt)]] void FunctionCEVal3(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEFloat)]] void FunctionCEVal4(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CETrue)]] void FunctionCEVal5(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEFalse)]] void FunctionCEVal6(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", CEEnum)]] void FunctionCEVal7(){}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] void FunctionCEVal8(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CENullptr)]] void FunctionCEVal9(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEStr)]] void FunctionCEVal10(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEInt)]] void FunctionCEVal11(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEFloat)]] void FunctionCEVal12(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CETrue)]] void FunctionCEVal13(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEFalse)]] void FunctionCEVal14(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", CEEnum)]] void FunctionCEVal15(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] void FunctionCEVal16(){}

[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, nullptr)]] void FunctionCEName1(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, "Text")]] void FunctionCEName2(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, 1)]] void FunctionCEName3(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, 3.14)]] void FunctionCEName4(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, true)]] void FunctionCEName5(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, false)]] void FunctionCEName6(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, TestEnum::EnumVal1)]] void FunctionCEName7(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] void FunctionCEName8(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] void FunctionCEName9(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, "Text")]] void FunctionCEName10(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, 1)]] void FunctionCEName11(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] void FunctionCEName12(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, true)]] void FunctionCEName13(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, false)]] void FunctionCEName14(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] void FunctionCEName15(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] void FunctionCEName16(){}

[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CENullptr)]] void FunctionCE1(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEStr)]] void FunctionCE2(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEInt)]] void FunctionCE3(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEFloat)]] void FunctionCE4(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CETrue)]] void FunctionCE5(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEFalse)]] void FunctionCE6(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEEnum)]] void FunctionCE7(){}
[[__sycl_detail__::add_ir_attributes_function(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] void FunctionCE8(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] void FunctionCE9(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] void FunctionCE10(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] void FunctionCE11(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] void FunctionCE12(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] void FunctionCE13(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] void FunctionCE14(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] void FunctionCE15(){}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] void FunctionCE16(){}

[[__sycl_detail__::add_ir_attributes_function("Attr1")]] void InvalidFunctionCEName1(){}                                               // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr, "Attr2")]] void InvalidFunctionCEName2(){}                             // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", nullptr)]] void InvalidFunctionCEName3(){}                             // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1")]] void InvalidFunctionCEName4(){}                           // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]] void InvalidFunctionCEName5(){}         // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]] void InvalidFunctionCEName6(){}         // expected-error {{attribute 'add_ir_attributes_function' must have an attribute value for each attribute name}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", {"Attr5", "Attr3"}, nullptr)]] void InvalidFunctionCEName7(){}                  // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", nullptr, {"Attr5", "Attr3"})]] void InvalidFunctionCEName8(){}                  // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]] void InvalidFunctionCEName9(){} // expected-error {{only the first argument of attribute 'add_ir_attributes_function' can be an initializer list}}

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr)]] GlobalVarStructLiteral1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Text")]] GlobalVarStructLiteral2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 1)]] GlobalVarStructLiteral3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", 3.14)]] GlobalVarStructLiteral4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", true)]] GlobalVarStructLiteral5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", false)]] GlobalVarStructLiteral6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", TestEnum::EnumVal1)]] GlobalVarStructLiteral7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] GlobalVarStructLiteral8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", nullptr)]] GlobalVarStructLiteral9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Text")]] GlobalVarStructLiteral10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", 1)]] GlobalVarStructLiteral11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", 3.14)]] GlobalVarStructLiteral12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", true)]] GlobalVarStructLiteral13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", false)]] GlobalVarStructLiteral14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] GlobalVarStructLiteral15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] GlobalVarStructLiteral16{};

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CENullptr)]] GlobalVarStructCEVal1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEStr)]] GlobalVarStructCEVal2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEInt)]] GlobalVarStructCEVal3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEFloat)]] GlobalVarStructCEVal4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CETrue)]] GlobalVarStructCEVal5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEFalse)]] GlobalVarStructCEVal6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", CEEnum)]] GlobalVarStructCEVal7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] GlobalVarStructCEVal8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CENullptr)]] GlobalVarStructCEVal9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEStr)]] GlobalVarStructCEVal10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEInt)]] GlobalVarStructCEVal11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEFloat)]] GlobalVarStructCEVal12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CETrue)]] GlobalVarStructCEVal13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEFalse)]] GlobalVarStructCEVal14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", CEEnum)]] GlobalVarStructCEVal15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] GlobalVarStructCEVal16{};

struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, nullptr)]] GlobalVarStructCEName1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, "Text")]] GlobalVarStructCEName2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, 1)]] GlobalVarStructCEName3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, 3.14)]] GlobalVarStructCEName4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, true)]] GlobalVarStructCEName5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, false)]] GlobalVarStructCEName6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, TestEnum::EnumVal1)]] GlobalVarStructCEName7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] GlobalVarStructCEName8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] GlobalVarStructCEName9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, "Text")]] GlobalVarStructCEName10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, 1)]] GlobalVarStructCEName11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] GlobalVarStructCEName12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, true)]] GlobalVarStructCEName13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, false)]] GlobalVarStructCEName14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] GlobalVarStructCEName15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] GlobalVarStructCEName16{};

struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CENullptr)]] GlobalVarStructCE1{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEStr)]] GlobalVarStructCE2{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEInt)]] GlobalVarStructCE3{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEFloat)]] GlobalVarStructCE4{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CETrue)]] GlobalVarStructCE5{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEFalse)]] GlobalVarStructCE6{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEEnum)]] GlobalVarStructCE7{};
struct [[__sycl_detail__::add_ir_attributes_global_variable(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] GlobalVarStructCE8{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] GlobalVarStructCE9{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] GlobalVarStructCE10{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] GlobalVarStructCE11{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] GlobalVarStructCE12{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] GlobalVarStructCE13{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] GlobalVarStructCE14{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] GlobalVarStructCE15{};
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] GlobalVarStructCE16{};

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1")]] InvalidGlobalVarStruct1{};                                               // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr, "Attr2")]] InvalidGlobalVarStruct2{};                             // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", nullptr)]] InvalidGlobalVarStruct3{};                             // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1")]] InvalidGlobalVarStruct4{};                           // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]] InvalidGlobalVarStruct5{};         // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]] InvalidGlobalVarStruct6{};         // expected-error {{attribute 'add_ir_attributes_global_variable' must have an attribute value for each attribute name}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", {"Attr5", "Attr3"}, nullptr)]] InvalidGlobalVarStruct7{};                  // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", nullptr, {"Attr5", "Attr3"})]] InvalidGlobalVarStruct8{};                  // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]] InvalidGlobalVarStruct9{}; // expected-error {{only the first argument of attribute 'add_ir_attributes_global_variable' can be an initializer list}}struct SpecialClassStructLiteral1{void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", nullptr)]] int x) {}};

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
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructLiteral16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] int x) {}
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
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEVal16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] int x) {}
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
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, nullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, "Text")]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, 1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, 3.14)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, true)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, false)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCEName16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1)]] int x) {}
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
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE9 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE10 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEStr)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE11 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEInt)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE12 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE13 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CETrue)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE14 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE15 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]] int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructCE16 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum)]] int x) {}
};

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
