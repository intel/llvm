// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

#include "Inputs/sycl.hpp"

__attribute__((sycl_global_var)) int GlobalWithAttribute;

__attribute__((sycl_global_var)) extern int ExternGlobalWithAttribute;

namespace NS {
    __attribute__((sycl_global_var)) int NSGlobalWithAttribute;
}

struct S {
    __attribute__((sycl_global_var)) static int StaticMember;

    // expected-error@+1 {{attribute only applies to global variables}}
    __attribute__((sycl_global_var)) int InstanceMember;
};
int S::StaticMember = 0;

__attribute__((sycl_global_var)) S GlobalStruct;

__attribute__((sycl_global_var)) static S StaticGlobal;

static union {
    // expected-error@+1 {{attribute only applies to global variables}}
    __attribute__((sycl_global_var)) int AnonymousStaticUnionInstanceMember;
};

// expected-error@+1 {{attribute takes no arguments}}
__attribute__((sycl_global_var(42))) int GlobalWithAttributeArg;

int GlobalNoAttribute;

// expected-error@+1 {{attribute only applies to global variables}}
__attribute__((sycl_global_var)) void F() {
    // expected-error@+1 {{attribute only applies to global variables}}
    __attribute__((sycl_global_var)) static int StaticLocalVar;

    // expected-error@+1 {{attribute only applies to global variables}}
    __attribute__((sycl_global_var)) int Local;

    cl::sycl::kernel_single_task<class kernel_name>([=] () {
        (void)GlobalWithAttribute;
        (void)ExternGlobalWithAttribute;
        (void)NS::NSGlobalWithAttribute;
        (void)S::StaticMember;
        (void)GlobalStruct.InstanceMember;
        (void)StaticGlobal.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)StaticLocalVar; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)AnonymousStaticUnionInstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)GlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@Inputs/sycl.hpp:* {{called by}}
    });
}
