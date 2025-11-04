// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -fsyntax-only -verify %s

// Tests that add_ir_attributes_* allows merging when the filter and IR
// attributes to generate are the same.

void FunctionRedecl1();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl1();  // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl1();
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]] void FunctionRedecl1();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, false)]] void FunctionRedecl1(); // expected-error {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]] void FunctionRedecl1(){};

[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl2();
void FunctionRedecl2();

void FunctionRedecl3();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl3();  // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl3();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]] void FunctionRedecl3();
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl3();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, false)]] void FunctionRedecl3(); // expected-error {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}} expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function({"Attr1"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl3();           // expected-error {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}} expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl3(){};                    // expected-error {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}}

void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]] void FunctionRedecl4();
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]] void FunctionRedecl4(){};

[[__sycl_detail__::add_ir_attributes_function({"Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl5();
void FunctionRedecl5();

void FunctionRedecl6();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl6();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl6();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]] void FunctionRedecl6();
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]] void FunctionRedecl6();
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr3", false)]] void FunctionRedecl6(){};

[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]]  // expected-error {{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, false)]] // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]]
void FunctionDecl1(){};

[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]]  // expected-error 3{{attribute '__sycl_detail__::add_ir_attributes_function' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, false)]] // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function({"Attr1"}, "Attr1", "Attr2", 1, true)]]           // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]                      // expected-note {{conflicting attribute is here}}
void FunctionDecl2(){};

[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function("Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_function("Attr3", false)]]
void FunctionDecl3(){};

[[__sycl_detail__::add_ir_attributes_function({"Attr3"}, "Attr1", "Attr2", 1, true)]]
void FunctionDecl4(){};

[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_function({"Attr1", "Attr3"}, "Attr3", false)]]
void FunctionDecl5(){};

struct GlobalVarStructRedecl1;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl1;  // expected-note {{conflicting attribute is here}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl1;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]] GlobalVarStructRedecl1;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, false)]] GlobalVarStructRedecl1; // expected-error {{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructRedecl1{};

struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl2;
struct GlobalVarStructRedecl2;

struct GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;  // expected-note {{conflicting attribute is here}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, false)]] GlobalVarStructRedecl3; // expected-error {{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}} expected-note {{conflicting attribute is here}}
struct [[__sycl_detail__::add_ir_attributes_global_variable({"Attr1"}, "Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3;           // expected-error {{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}} expected-note {{conflicting attribute is here}}
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl3{};                    // expected-error {{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}}

struct GlobalVarStructRedecl4;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl4;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructRedecl4;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]] GlobalVarStructRedecl4;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructRedecl4{};

struct GlobalVarStructRedecl5;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] void GlobalVarStructRedecl5;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] void GlobalVarStructRedecl5;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]] void GlobalVarStructRedecl5;
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] void GlobalVarStructRedecl5{};

[[__sycl_detail__::add_ir_attributes_global_variable({"Attr3"}, "Attr1", "Attr2", 1, true)]] struct GlobalVarStructRedecl6;
struct GlobalVarStructRedecl6{};

struct
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]]  // expected-error {{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, false)]] // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]]
GlobalVarStructDecl1{};

struct
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]  // expected-error 3{{attribute '__sycl_detail__::add_ir_attributes_global_variable' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, false)]] // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1"}, "Attr1", "Attr2", 1, true)]]           // expected-note {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]                      // expected-note {{conflicting attribute is here}}
GlobalVarStructDecl2{};

struct
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]]
GlobalVarStructDecl3{};

struct
[[__sycl_detail__::add_ir_attributes_function({"Attr3"}, "Attr1", "Attr2", 1, true)]]
GlobalVarStructDecl4{};

struct
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr3", "Attr1"}, "Attr1", "Attr2", 1, true)]]
[[__sycl_detail__::add_ir_attributes_global_variable({"Attr1", "Attr3"}, "Attr3", false)]]
GlobalVarStructDecl5{};

struct GlobalVarStructBase {};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr1", "Attr2", 1, true)]] GlobalVarStructInherit1 : GlobalVarStructBase{};
struct GlobalVarStructInherit2 : GlobalVarStructInherit1 {};
struct [[__sycl_detail__::add_ir_attributes_global_variable("Attr3", false)]] GlobalVarStructInherit3 : GlobalVarStructInherit1{};

struct __attribute__((sycl_special_class)) SpecialClassStructBase {
  virtual void __init(int x) {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructInherit1 : SpecialClassStructBase {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, true)]] int x) override {}
};
struct __attribute__((sycl_special_class)) SpecialClassStructInherit2 : SpecialClassStructInherit1 {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", false)]] int x) override {}
};

struct __attribute__((sycl_special_class)) SpecialClassStruct1 {
  virtual void __init(
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, true)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr2", "Attr1", true, 1)]]  // expected-error {{attribute '__sycl_detail__::add_ir_attributes_kernel_parameter' is already applied with different arguments}}
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, false)]] // expected-note {{conflicting attribute is here}}
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", false)]]
    int x) {}
};

struct __attribute__((sycl_special_class)) SpecialClassStruct2 {
  virtual void __init(
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr1", "Attr2", 1, true)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr2", "Attr1", true, 1)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter("Attr3", false)]]
    int x) {}
};

struct __attribute__((sycl_special_class)) SpecialClassStruct3 {
  virtual void __init(
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr1", "Attr2", 1, true)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr2", "Attr1", true, 1)]]
    [[__sycl_detail__::add_ir_attributes_kernel_parameter({"Attr1", "Attr3"}, "Attr3", false)]]
    int x) {}
};
