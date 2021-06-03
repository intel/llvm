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
__attribute__((sycl_global_var)) void HppF() {
  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) static int StaticLocalVar;

  // expected-error@+1 {{attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Local;

  cl::sycl::kernel_single_task<class kernel_name>([=] () {
    (void)HppGlobalWithAttribute;
    (void)HppExternGlobalWithAttribute;
    (void)NS::HppNSGlobalWithAttribute;
    (void)HppS::StaticMember;
    (void)HppGlobalStruct.InstanceMember;
    (void)HppStaticGlobal.InstanceMember;
    (void)HppStructTemplate<int>::StaticMember;
    (void)HppGlobalWithAttrMacro;
    (void)HppGlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@Inputs/sycl.hpp:* {{called by}}
  });
}

# 69 "header.hpp" 2 // Return from the simulated #include (with the last line number of the "header.hpp" file)

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
__attribute__((sycl_global_var)) int GlobalWithAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
__attribute__((sycl_global_var)) extern int ExternGlobalWithAttribute;

namespace NS {
  // expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
  __attribute__((sycl_global_var)) int NSGlobalWithAttribute;
}

struct S {
  // expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
  __attribute__((sycl_global_var)) static int StaticMember;

  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int InstanceMember;
};
int S::StaticMember = 0;

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
__attribute__((sycl_global_var)) S GlobalStruct;

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
__attribute__((sycl_global_var)) static S StaticGlobal;

static union {
  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int AnonymousStaticUnionInstanceMember;
};

// expected-error@+1 {{attribute takes no arguments}}
__attribute__((sycl_global_var(42))) int GlobalWithAttributeArg;

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
__attribute__((sycl_global_var)) HppStructTemplate<int> GlobalTemplateStructWithAttribute;
HppStructTemplate<int> GlobalTemplateStructNoAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only applies to system header global variables}}
SYCLGLOBALVAR_ATTR_MACRO int GlobalWithAttrMacro;

int GlobalNoAttribute;

// expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
__attribute__((sycl_global_var)) void F() {
  // expected-error@+2 {{'sycl_global_var' attribute only applies to system header global variables}}
  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) static int StaticLocalVar;

  // expected-error@+1 {{'sycl_global_var' attribute only applies to global variables}}
  __attribute__((sycl_global_var)) int Local;

  cl::sycl::kernel_single_task<class kernel_name>([=] () {
    (void)HppGlobalWithAttribute;
    (void)HppExternGlobalWithAttribute;
    (void)NS::HppNSGlobalWithAttribute;
    (void)HppS::StaticMember;
    (void)HppGlobalStruct.InstanceMember;
    (void)HppStaticGlobal.InstanceMember;
    (void)HppStructTemplate<int>::StaticMember;
    (void)GlobalTemplateStructNoAttribute.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const global variable}}
    (void)GlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@Inputs/sycl.hpp:* {{called by}}
  });
}
