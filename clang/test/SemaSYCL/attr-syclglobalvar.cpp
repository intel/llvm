// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

#include "Inputs/sycl.hpp"

# 5 "header.hpp" 1 3 // Simulate a system #include to enter new file named header.hpp at line 5

#define SYCLGLOBALVAR_ATTR_MACRO __attribute__((sycl_global_var))

__attribute__((sycl_global_var)) int HppGlobalWithAttribute;

__attribute__((sycl_global_var)) extern int HppExternGlobalWithAttribute;

namespace NS {
  __attribute__((sycl_global_var)) int HppNSGlobalWithAttribute;
}

struct HppS {
  __attribute__((sycl_global_var)) static int StaticMember;

  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int InstanceMember;
};
int HppS::StaticMember = 0;

__attribute__((sycl_global_var)) HppS HppGlobalStruct;

__attribute__((sycl_global_var)) static HppS HppStaticGlobal;

static union {
  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int HppAnonymousStaticUnionInstanceMember;
};

// expected-error@+1 {{attribute takes no arguments}}
__attribute__((sycl_global_var(42))) int HppGlobalWithAttributeArg;

template<typename T> struct HppStructTemplate {
  __attribute__((sycl_global_var)) static T StaticMember;

  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int InstanceMember;
};

SYCLGLOBALVAR_ATTR_MACRO int HppGlobalWithAttrMacro;

int HppGlobalNoAttribute;

// expected-error@+1 {{attribute only applies to global variables}}
__attribute__((sycl_global_var)) void HppF(
  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Param
) {
  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) static int StaticLocalVar;

  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Local;

  sycl::kernel_single_task<class kernel_name>([=] () {
    (void)HppGlobalWithAttribute; // ok
    (void)HppExternGlobalWithAttribute; // ok
    (void)NS::HppNSGlobalWithAttribute; // ok
    (void)HppS::StaticMember; // ok
    (void)HppGlobalStruct.InstanceMember; // ok
    (void)HppStaticGlobal.InstanceMember; // ok
    (void)HppAnonymousStaticUnionInstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    (void)HppGlobalWithAttributeArg; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppStructTemplate<int>::StaticMember; // ok
    (void)HppGlobalWithAttrMacro; // ok
    (void)HppGlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernel_name, (lambda at}}
  });
}

# 74 "header.hpp" 2 // Return from the simulated #include (with the last line number of the "header.hpp" file)

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
__attribute__((sycl_global_var)) int CppGlobalWithAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
__attribute__((sycl_global_var)) extern int CppExternGlobalWithAttribute;

namespace NS {
  // expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
  __attribute__((sycl_global_var)) int CppNSGlobalWithAttribute;
}

struct CppS {
  // expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
  __attribute__((sycl_global_var)) static int StaticMember;

  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int InstanceMember;
};
int CppS::StaticMember = 0;

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
__attribute__((sycl_global_var)) CppS CppGlobalStruct;

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
__attribute__((sycl_global_var)) static CppS CppStaticGlobal;

static union {
  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int CppAnonymousStaticUnionInstanceMember;
};

// expected-error@+1 {{attribute takes no arguments}}
__attribute__((sycl_global_var(42))) int CppGlobalWithAttributeArg;

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
__attribute__((sycl_global_var)) HppStructTemplate<int> CppGlobalTemplateStructWithAttribute;
HppStructTemplate<int> CppGlobalTemplateStructNoAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only supported within a system header}}
SYCLGLOBALVAR_ATTR_MACRO int CppGlobalWithAttrMacro;

int GlobalNoAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
__attribute__((sycl_global_var)) void F(
  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Param
) {
  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) static int StaticLocalVar;

  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Local;

  sycl::kernel_single_task<class kernel_name>([=] () {
    (void)HppGlobalWithAttribute; // ok
    (void)CppGlobalWithAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppExternGlobalWithAttribute; // ok
    (void)CppExternGlobalWithAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)NS::HppNSGlobalWithAttribute; // ok
    (void)NS::CppNSGlobalWithAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppS::StaticMember; // ok
    (void)CppS::StaticMember; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppGlobalStruct.InstanceMember; // ok
    (void)CppGlobalStruct.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppStaticGlobal.InstanceMember; // ok
    (void)CppStaticGlobal.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    (void)CppAnonymousStaticUnionInstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    (void)CppGlobalWithAttributeArg; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)HppStructTemplate<int>::StaticMember; // ok
    (void)CppGlobalTemplateStructWithAttribute.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)CppGlobalTemplateStructNoAttribute.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)GlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<kernel_name, (lambda at}}
  });
}
