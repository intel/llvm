// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -isystem %S %s

#include <Inputs-isystem/attr-syclglobalvar.hpp>
#include "../Inputs/sycl.hpp"

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
    (void)GlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@../Inputs/sycl.hpp:* {{called by}}
  });
}
