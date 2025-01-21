// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// Tests valid and invalid arguments in
// __sycl_detail__::add_ir_annotations_member attributes.

enum TestEnum {
  EnumVal1,
  EnumVal2
};

enum class ScopedTestEnum : short {
  ScopedEnumVal1,
  ScopedEnumVal2
};

constexpr decltype(nullptr) CENullptr = nullptr;
constexpr const char CEStr[] = "Text";
constexpr int CEInt = 1;
constexpr float CEFloat = 3.14;
constexpr bool CETrue = true;
constexpr bool CEFalse = false;
constexpr TestEnum CEEnum = TestEnum::EnumVal1;
constexpr char CEChar = 'F';
constexpr ScopedTestEnum CESEnum = ScopedTestEnum::ScopedEnumVal2;

constexpr const char CEAttrName1[] = "CEAttr1";
constexpr const char CEAttrName2[] = "CEAttr2";
constexpr const char CEAttrName3[] = "CEAttr3";
constexpr const char CEAttrName4[] = "CEAttr4";
constexpr const char CEAttrName5[] = "CEAttr5";
constexpr const char CEAttrName6[] = "CEAttr6";
constexpr const char CEAttrName7[] = "CEAttr7";
constexpr const char CEAttrName8[] = "CEAttr8";
constexpr const char CEAttrName9[] = "CEAttr9";

struct ClassWithAnnotFieldLiteral1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", nullptr)]];
};
struct ClassWithAnnotFieldLiteral2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Text")]];
};
struct ClassWithAnnotFieldLiteral3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", 1)]];
};
struct ClassWithAnnotFieldLiteral4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", 3.14)]];
};
struct ClassWithAnnotFieldLiteral5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", true)]];
};
struct ClassWithAnnotFieldLiteral6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", false)]];
};
struct ClassWithAnnotFieldLiteral7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", TestEnum::EnumVal1)]];
};
struct ClassWithAnnotFieldLiteral8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", 'F')]];
};
struct ClassWithAnnotFieldLiteral9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", "Attr9", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldLiteral10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", nullptr)]];
};
struct ClassWithAnnotFieldLiteral11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", "Text")]];
};
struct ClassWithAnnotFieldLiteral12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", 1)]];
};
struct ClassWithAnnotFieldLiteral13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", 3.14)]];
};
struct ClassWithAnnotFieldLiteral14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", true)]];
};
struct ClassWithAnnotFieldLiteral15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", false)]];
};
struct ClassWithAnnotFieldLiteral16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", TestEnum::EnumVal1)]];
};
struct ClassWithAnnotFieldLiteral17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", 'F')]];
};
struct ClassWithAnnotFieldLiteral18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", "Attr9", nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldLiteral19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldLiteral20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", ScopedTestEnum::ScopedEnumVal2)]];
};

struct ClassWithAnnotFieldCEVal1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CENullptr)]];
};
struct ClassWithAnnotFieldCEVal2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEStr)]];
};
struct ClassWithAnnotFieldCEVal3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEInt)]];
};
struct ClassWithAnnotFieldCEVal4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEFloat)]];
};
struct ClassWithAnnotFieldCEVal5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CETrue)]];
};
struct ClassWithAnnotFieldCEVal6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEFalse)]];
};
struct ClassWithAnnotFieldCEVal7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEEnum)]];
};
struct ClassWithAnnotFieldCEVal8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEChar)]];
};
struct ClassWithAnnotFieldCEVal9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", "Attr9", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]];
};
struct ClassWithAnnotFieldCEVal10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CENullptr)]];
};
struct ClassWithAnnotFieldCEVal11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEStr)]];
};
struct ClassWithAnnotFieldCEVal12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEInt)]];
};
struct ClassWithAnnotFieldCEVal13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEFloat)]];
};
struct ClassWithAnnotFieldCEVal14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CETrue)]];
};
struct ClassWithAnnotFieldCEVal15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEFalse)]];
};
struct ClassWithAnnotFieldCEVal16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEEnum)]];
};
struct ClassWithAnnotFieldCEVal17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CEChar)]];
};
struct ClassWithAnnotFieldCEVal18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", "Attr8", "Attr9", CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]];
};
struct ClassWithAnnotFieldCEVal19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CESEnum)]];
};
struct ClassWithAnnotFieldCEVal20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", CESEnum)]];
};

struct ClassWithAnnotFieldCEName1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, nullptr)]];
};
struct ClassWithAnnotFieldCEName2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, "Text")]];
};
struct ClassWithAnnotFieldCEName3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, 1)]];
};
struct ClassWithAnnotFieldCEName4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, 3.14)]];
};
struct ClassWithAnnotFieldCEName5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, true)]];
};
struct ClassWithAnnotFieldCEName6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, false)]];
};
struct ClassWithAnnotFieldCEName7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, TestEnum::EnumVal1)]];
};
struct ClassWithAnnotFieldCEName8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, 'F')]];
};
struct ClassWithAnnotFieldCEName9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldCEName10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, nullptr)]];
};
struct ClassWithAnnotFieldCEName11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, "Text")]];
};
struct ClassWithAnnotFieldCEName12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, 1)]];
};
struct ClassWithAnnotFieldCEName13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, 3.14)]];
};
struct ClassWithAnnotFieldCEName14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, true)]];
};
struct ClassWithAnnotFieldCEName15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, false)]];
};
struct ClassWithAnnotFieldCEName16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, TestEnum::EnumVal1)]];
};
struct ClassWithAnnotFieldCEName17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, 'F')]];
};
struct ClassWithAnnotFieldCEName18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldCEName19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, ScopedTestEnum::ScopedEnumVal2)]];
};
struct ClassWithAnnotFieldCEName20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, ScopedTestEnum::ScopedEnumVal2)]];
};

struct ClassWithAnnotFieldCE1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CENullptr)]];
};
struct ClassWithAnnotFieldCE2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEStr)]];
};
struct ClassWithAnnotFieldCE3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEInt)]];
};
struct ClassWithAnnotFieldCE4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEFloat)]];
};
struct ClassWithAnnotFieldCE5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CETrue)]];
};
struct ClassWithAnnotFieldCE6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEFalse)]];
};
struct ClassWithAnnotFieldCE7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEEnum)]];
};
struct ClassWithAnnotFieldCE8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEChar)]];
};
struct ClassWithAnnotFieldCE9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]];
};
struct ClassWithAnnotFieldCE10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CENullptr)]];
};
struct ClassWithAnnotFieldCE11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEStr)]];
};
struct ClassWithAnnotFieldCE12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEInt)]];
};
struct ClassWithAnnotFieldCE13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEFloat)]];
};
struct ClassWithAnnotFieldCE14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CETrue)]];
};
struct ClassWithAnnotFieldCE15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEFalse)]];
};
struct ClassWithAnnotFieldCE16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEEnum)]];
};
struct ClassWithAnnotFieldCE17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEChar)]];
};
struct ClassWithAnnotFieldCE18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]];
};
struct ClassWithAnnotFieldCE19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CESEnum)]];
};
struct ClassWithAnnotFieldCE20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, CEAttrName1, CESEnum)]];
};

template <decltype(nullptr) Null> struct ClassWithAnnotFieldTemplate1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", Null)]];
};
template <const char *Str> struct ClassWithAnnotFieldTemplate2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", Str)]];
};
template <int I> struct ClassWithAnnotFieldTemplate3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", I)]];
};
template <bool B> struct ClassWithAnnotFieldTemplate4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", B)]];
};
template <TestEnum E> struct ClassWithAnnotFieldTemplate5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", E)]];
};
template <char C> struct ClassWithAnnotFieldTemplate6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", C)]];
};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C, ScopedTestEnum SE> struct ClassWithAnnotFieldTemplate7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", "Attr4", "Attr5", "Attr6", "Attr7", Null, Str, I, B, E, C, SE)]];
};
template <decltype(nullptr) Null> struct ClassWithAnnotFieldTemplate8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, Null)]];
};
template <const char *Str> struct ClassWithAnnotFieldTemplate9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, Str)]];
};
template <int I> struct ClassWithAnnotFieldTemplate10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, I)]];
};
template <bool B> struct ClassWithAnnotFieldTemplate11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, B)]];
};
template <TestEnum E> struct ClassWithAnnotFieldTemplate12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, E)]];
};
template <char C> struct ClassWithAnnotFieldTemplate13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, C)]];
};
template <decltype(nullptr) Null, const char *Str, int I, bool B, TestEnum E, char C, ScopedTestEnum SE> struct ClassWithAnnotFieldTemplate14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, Null, Str, I, B, E, C, SE)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, nullptr)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, "Text")]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, 1)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, 3.14)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, true)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, false)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate21 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, TestEnum::EnumVal1)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate22 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, 'F')]];
};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8, const char *Name9> struct ClassWithAnnotFieldTemplate23 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, Name9, nullptr, "Text", 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate24 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CENullptr)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate25 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEStr)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate26 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEInt)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate27 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEFloat)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate28 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CETrue)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate29 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEFalse)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate30 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEEnum)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate31 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CEChar)]];
};
template <const char *Name1, const char *Name2, const char *Name3, const char *Name4, const char *Name5, const char *Name6, const char *Name7, const char *Name8, const char *Name9> struct ClassWithAnnotFieldTemplate32 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name1, Name2, Name3, Name4, Name5, Name6, Name7, Name8, Name9, CENullptr, CEStr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]];
};
template <ScopedTestEnum SE> struct ClassWithAnnotFieldTemplate33 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", SE)]];
};
template <ScopedTestEnum SE> struct ClassWithAnnotFieldTemplate34 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEAttrName1, SE)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate35 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, ScopedTestEnum::ScopedEnumVal2)]];
};
template <const char *Name> struct ClassWithAnnotFieldTemplate36 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Name, CESEnum)]];
};
void InstantiateClassWithAnnotFieldTemplates() {
  ClassWithAnnotFieldTemplate1<nullptr> InstantiatedCWAFS1;
  ClassWithAnnotFieldTemplate1<CENullptr> InstantiatedCWAFS2;
  ClassWithAnnotFieldTemplate2<CEStr> InstantiatedCWAFS3;
  ClassWithAnnotFieldTemplate3<1> InstantiatedCWAFS4;
  ClassWithAnnotFieldTemplate3<CEInt> InstantiatedCWAFS5;
  ClassWithAnnotFieldTemplate4<true> InstantiatedCWAFS6;
  ClassWithAnnotFieldTemplate4<CETrue> InstantiatedCWAFS7;
  ClassWithAnnotFieldTemplate4<false> InstantiatedCWAFS8;
  ClassWithAnnotFieldTemplate4<CEFalse> InstantiatedCWAFS9;
  ClassWithAnnotFieldTemplate5<TestEnum::EnumVal1> InstantiatedCWAFS10;
  ClassWithAnnotFieldTemplate5<CEEnum> InstantiatedCWAFS11;
  ClassWithAnnotFieldTemplate6<'F'> InstantiatedCWAFS12;
  ClassWithAnnotFieldTemplate6<CEChar> InstantiatedCWAFS13;
  ClassWithAnnotFieldTemplate7<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2> InstantiatedCWAFS14;
  ClassWithAnnotFieldTemplate7<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar, CESEnum> InstantiatedCWAFS15;
  ClassWithAnnotFieldTemplate8<nullptr> InstantiatedCWAFS16;
  ClassWithAnnotFieldTemplate8<CENullptr> InstantiatedCWAFS17;
  ClassWithAnnotFieldTemplate9<CEStr> InstantiatedCWAFS18;
  ClassWithAnnotFieldTemplate10<1> InstantiatedCWAFS19;
  ClassWithAnnotFieldTemplate10<CEInt> InstantiatedCWAFS20;
  ClassWithAnnotFieldTemplate11<true> InstantiatedCWAFS21;
  ClassWithAnnotFieldTemplate11<CETrue> InstantiatedCWAFS22;
  ClassWithAnnotFieldTemplate11<false> InstantiatedCWAFS23;
  ClassWithAnnotFieldTemplate11<CEFalse> InstantiatedCWAFS24;
  ClassWithAnnotFieldTemplate12<TestEnum::EnumVal1> InstantiatedCWAFS25;
  ClassWithAnnotFieldTemplate12<CEEnum> InstantiatedCWAFS26;
  ClassWithAnnotFieldTemplate13<'F'> InstantiatedCWAFS27;
  ClassWithAnnotFieldTemplate13<CEChar> InstantiatedCWAFS28;
  ClassWithAnnotFieldTemplate14<nullptr, CEStr, 1, true, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2> InstantiatedCWAFS29;
  ClassWithAnnotFieldTemplate14<CENullptr, CEStr, CEInt, CETrue, CEEnum, CEChar, CESEnum> InstantiatedCWAFS30;
  ClassWithAnnotFieldTemplate15<CEAttrName1> InstantiatedCWAFS31;
  ClassWithAnnotFieldTemplate16<CEAttrName1> InstantiatedCWAFS32;
  ClassWithAnnotFieldTemplate17<CEAttrName1> InstantiatedCWAFS33;
  ClassWithAnnotFieldTemplate18<CEAttrName1> InstantiatedCWAFS34;
  ClassWithAnnotFieldTemplate19<CEAttrName1> InstantiatedCWAFS35;
  ClassWithAnnotFieldTemplate20<CEAttrName1> InstantiatedCWAFS36;
  ClassWithAnnotFieldTemplate21<CEAttrName1> InstantiatedCWAFS37;
  ClassWithAnnotFieldTemplate22<CEAttrName1> InstantiatedCWAFS38;
  ClassWithAnnotFieldTemplate23<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9> InstantiatedCWAFS39;
  ClassWithAnnotFieldTemplate24<CEAttrName1> InstantiatedCWAFS40;
  ClassWithAnnotFieldTemplate25<CEAttrName1> InstantiatedCWAFS41;
  ClassWithAnnotFieldTemplate26<CEAttrName1> InstantiatedCWAFS42;
  ClassWithAnnotFieldTemplate27<CEAttrName1> InstantiatedCWAFS43;
  ClassWithAnnotFieldTemplate28<CEAttrName1> InstantiatedCWAFS44;
  ClassWithAnnotFieldTemplate29<CEAttrName1> InstantiatedCWAFS45;
  ClassWithAnnotFieldTemplate30<CEAttrName1> InstantiatedCWAFS46;
  ClassWithAnnotFieldTemplate31<CEAttrName1> InstantiatedCWAFS47;
  ClassWithAnnotFieldTemplate32<CEAttrName1, CEAttrName2, CEAttrName3, CEAttrName4, CEAttrName5, CEAttrName6, CEAttrName7, CEAttrName8, CEAttrName9> InstantiatedCWAFS48;
  ClassWithAnnotFieldTemplate33<ScopedTestEnum::ScopedEnumVal2> InstantiatedCWAFS49;
  ClassWithAnnotFieldTemplate34<ScopedTestEnum::ScopedEnumVal2> InstantiatedCWAFS50;
  ClassWithAnnotFieldTemplate35<CEAttrName1> InstantiatedCWAFS51;
  ClassWithAnnotFieldTemplate36<CEAttrName1> InstantiatedCWAFS52;

  (void)*InstantiatedCWAFS1.ptr;
  (void)*InstantiatedCWAFS2.ptr;
  (void)*InstantiatedCWAFS3.ptr;
  (void)*InstantiatedCWAFS4.ptr;
  (void)*InstantiatedCWAFS5.ptr;
  (void)*InstantiatedCWAFS6.ptr;
  (void)*InstantiatedCWAFS7.ptr;
  (void)*InstantiatedCWAFS8.ptr;
  (void)*InstantiatedCWAFS9.ptr;
  (void)*InstantiatedCWAFS10.ptr;
  (void)*InstantiatedCWAFS11.ptr;
  (void)*InstantiatedCWAFS12.ptr;
  (void)*InstantiatedCWAFS13.ptr;
  (void)*InstantiatedCWAFS14.ptr;
  (void)*InstantiatedCWAFS15.ptr;
  (void)*InstantiatedCWAFS16.ptr;
  (void)*InstantiatedCWAFS17.ptr;
  (void)*InstantiatedCWAFS18.ptr;
  (void)*InstantiatedCWAFS19.ptr;
  (void)*InstantiatedCWAFS20.ptr;
  (void)*InstantiatedCWAFS21.ptr;
  (void)*InstantiatedCWAFS22.ptr;
  (void)*InstantiatedCWAFS23.ptr;
  (void)*InstantiatedCWAFS24.ptr;
  (void)*InstantiatedCWAFS25.ptr;
  (void)*InstantiatedCWAFS26.ptr;
  (void)*InstantiatedCWAFS27.ptr;
  (void)*InstantiatedCWAFS28.ptr;
  (void)*InstantiatedCWAFS29.ptr;
  (void)*InstantiatedCWAFS30.ptr;
  (void)*InstantiatedCWAFS31.ptr;
  (void)*InstantiatedCWAFS32.ptr;
  (void)*InstantiatedCWAFS33.ptr;
  (void)*InstantiatedCWAFS34.ptr;
  (void)*InstantiatedCWAFS35.ptr;
  (void)*InstantiatedCWAFS36.ptr;
}

struct InvalidClassWithAnnotField1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1")]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", nullptr, "Attr2")]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", nullptr)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr5", "Attr3"}, "Attr1")]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr5", "Attr3"}, "Attr1", nullptr, "Attr2")]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr5", "Attr3"}, "Attr1", "Attr2", nullptr)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
struct InvalidClassWithAnnotField7 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", {"Attr5", "Attr3"}, nullptr)]]; // expected-error {{only the first argument of attribute 'add_ir_annotations_member' can be an initializer list}}
};
struct InvalidClassWithAnnotField8 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", nullptr, {"Attr5", "Attr3"})]]; // expected-error {{only the first argument of attribute 'add_ir_annotations_member' can be an initializer list}}
};
struct InvalidClassWithAnnotField9 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", {"Attr5", "Attr3"}, nullptr, "Text")]]; // expected-error {{only the first argument of attribute 'add_ir_annotations_member' can be an initializer list}}
};
struct InvalidClassWithAnnotField10 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({1}, "Attr1", nullptr)]]; // expected-error {{initializer list in the first argument of 'add_ir_annotations_member' must contain only string literals}}
};
struct InvalidClassWithAnnotField11 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({true, "Attr3"}, "Attr1", nullptr)]]; // expected-error {{initializer list in the first argument of 'add_ir_annotations_member' must contain only string literals}}
};
struct InvalidClassWithAnnotField12 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr3", 'c'}, "Attr1", nullptr)]]; // expected-error {{initializer list in the first argument of 'add_ir_annotations_member' must contain only string literals}}
};
struct InvalidClassWithAnnotField13 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(nullptr, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField14 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(1, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField15 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(3.14, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField16 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(true, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField17 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(false, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField18 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(TestEnum::EnumVal1, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField19 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member('F', "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField20 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2, nullptr, 1, 3.14, true, false, TestEnum::EnumVal1, 'F', ScopedTestEnum::ScopedEnumVal2)]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField21 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", 3.14, "Attr3", 1, 3.14, true)]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField22 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CENullptr, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField23 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEInt, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField24 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEFloat, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField25 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CETrue, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField26 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEFalse, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField27 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEEnum, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField28 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CEChar, "Attr1")]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField29 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum, CENullptr, CEInt, CEFloat, CETrue, CEFalse, CEEnum, CEChar, CESEnum)]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField30 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", CEFloat, "Attr3", CEInt, CEFloat, CETrue)]]; // expected-error {{each name argument in 'add_ir_annotations_member' must be a 'const char *' usable in a constant expression}}
};
struct InvalidClassWithAnnotField31 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", &CEInt)]]; // expected-error {{each value argument in 'add_ir_annotations_member' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
};
struct InvalidClassWithAnnotField32 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", 1, &CEInt, CEInt)]]; // expected-error {{each value argument in 'add_ir_annotations_member' must be an integer, a floating point, a character, a boolean, 'const char *', or an enumerator usable as a constant expression}}
};

struct [[__sycl_detail__::add_ir_annotations_member("Attr1", 1)]] InvalidAnnotationsMemberSubjectStruct;     // expected-error {{'add_ir_annotations_member' attribute only applies to non-static data members}}
[[__sycl_detail__::add_ir_annotations_member("Attr1", 1)]] void InvalidAnnotationsMemberSubjectFunction() {} // expected-error {{'add_ir_annotations_member' attribute only applies to non-static data members}}
[[__sycl_detail__::add_ir_annotations_member("Attr1", 1)]] int InvalidAnnotationsMemberSubjectVar;           // expected-error {{'add_ir_annotations_member' attribute only applies to non-static data members}}
