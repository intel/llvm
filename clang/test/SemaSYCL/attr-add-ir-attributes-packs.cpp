// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -fsyntax-only -verify %s

constexpr const char AttrName1[] = "Attr1";
constexpr const char AttrName2[] = "Attr2";
constexpr const char AttrName3[] = "Attr3";
constexpr const char AttrVal1[] = "Val1";
constexpr const char AttrVal2[] = "Val2";
constexpr const char AttrVal3[] = "Val3";

template <int... Is> [[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", "Attr3", Is...)]] void FunctionTemplate1() {}                     // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}
template <int... Is> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] void FunctionTemplate2() {} // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}
template <const char *...Names> [[__sycl_detail__::add_ir_attributes_function(Names..., 1, 2, 3)]] void FunctionTemplate3() {}                         // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}
template <const char *...Names> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] void FunctionTemplate4() {}     // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}
template <const char *...Strs> [[__sycl_detail__::add_ir_attributes_function(Strs...)]] void FunctionTemplate5() {}                                    // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}
template <const char *...Strs> [[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, Strs...)]] void FunctionTemplate6() {}                // expected-error {{attribute 'add_ir_attributes_function' must specify a value for each specified name in the argument list}}

void InstantiateFunctionTemplates() {
  FunctionTemplate1<1, 2, 3>();
  FunctionTemplate1<1, 2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate1<1, 2>' requested here}}

  FunctionTemplate2<1, 2, 3>();
  FunctionTemplate2<1, 2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate2<1, 2>' requested here}}

  FunctionTemplate3<AttrName1, AttrName2, AttrName3>();
  FunctionTemplate3<AttrName1, AttrName2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate3<AttrName1, AttrName2>' requested here}}

  FunctionTemplate4<AttrName1, AttrName2, AttrName3>();
  FunctionTemplate4<AttrName1, AttrName2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate4<AttrName1, AttrName2>' requested here}}

  FunctionTemplate5<AttrName1, AttrVal1>();
  FunctionTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2>();
  FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3>();
  FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  FunctionTemplate6<AttrName1, AttrVal1>();
  FunctionTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2>();
  FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3>();
  FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>(); // expected-note {{in instantiation of function template specialization 'FunctionTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}
}

template <int... Is> struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", "Attr3", Is...)]] GlobalVarStructTemplate1{};                     // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}
template <int... Is> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] GlobalVarStructTemplate2{}; // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}
template <const char *...Names> struct [[__sycl_detail__::add_ir_attributes_global_variable(Names..., 1, 2, 3)]] GlobalVarStructTemplate3{};                         // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}
template <const char *...Names> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] GlobalVarStructTemplate4{};     // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}
template <const char *...Strs> struct [[__sycl_detail__::add_ir_attributes_global_variable(Strs...)]] GlobalVarStructTemplate5{};                                    // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}
template <const char *...Strs> struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, Strs...)]] GlobalVarStructTemplate6{};                // expected-error {{attribute 'add_ir_attributes_global_variable' must specify a value for each specified name in the argument list}}

GlobalVarStructTemplate1<1, 2, 3> InstantiatedGV1;
GlobalVarStructTemplate1<1, 2> InstantiatedGV2; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate1<1, 2>' requested here}}

GlobalVarStructTemplate2<1, 2, 3> InstantiatedGV3;
GlobalVarStructTemplate2<1, 2> InstantiatedGV4; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate2<1, 2>' requested here}}

GlobalVarStructTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedGV5;
GlobalVarStructTemplate3<AttrName1, AttrName2> InstantiatedGV6; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate3<AttrName1, AttrName2>' requested here}}

GlobalVarStructTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedGV7;
GlobalVarStructTemplate4<AttrName1, AttrName2> InstantiatedGV8; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate4<AttrName1, AttrName2>' requested here}}

GlobalVarStructTemplate5<AttrName1, AttrVal1> InstantiatedGV9;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedGV10;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedGV11;
GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedGV12; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

GlobalVarStructTemplate6<AttrName1, AttrVal1> InstantiatedGV13;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedGV14;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedGV15;
GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedGV16; // expected-note {{in instantiation of template class 'GlobalVarStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

template <int... Is> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", "Attr3", Is...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};
template <int... Is> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate2 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};
template <const char *...Names> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate3 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Names..., 1, 2, 3)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};
template <const char *...Names> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate4 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, Names..., 1, 2, 3)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};
template <const char *...Strs> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate5 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(Strs...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};
template <const char *...Strs> struct __attribute__((sycl_special_class)) SpecialClassStructTemplate6 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, Strs...)]] int x) {} // expected-error {{attribute 'add_ir_attributes_kernel_parameter' must specify a value for each specified name in the argument list}}
};

void InstantiateSpecialClassStructTemplates() {
  SpecialClassStructTemplate1<1, 2, 3> InstantiatedSCS1;
  SpecialClassStructTemplate1<1, 2> InstantiatedSCS2; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate1<1, 2>' requested here}}

  SpecialClassStructTemplate2<1, 2, 3> InstantiatedSCS3;
  SpecialClassStructTemplate2<1, 2> InstantiatedSCS4; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate2<1, 2>' requested here}}

  SpecialClassStructTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedSCS5;
  SpecialClassStructTemplate3<AttrName1, AttrName2> InstantiatedSCS6; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate3<AttrName1, AttrName2>' requested here}}

  SpecialClassStructTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedSCS7;
  SpecialClassStructTemplate4<AttrName1, AttrName2> InstantiatedSCS8; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate4<AttrName1, AttrName2>' requested here}}

  SpecialClassStructTemplate5<AttrName1, AttrVal1> InstantiatedSCS9;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedSCS10;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedSCS11;
  SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedSCS12; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  SpecialClassStructTemplate6<AttrName1, AttrVal1> InstantiatedSCS13;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedSCS14;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedSCS15;
  SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedSCS16; // expected-note {{in instantiation of template class 'SpecialClassStructTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}
}
