// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

#include "Inputs/sycl.hpp"

__attribute__((sycl_global_var)) int GlobalWithAttribute;

int GlobalNoAttribute;

// expected-error@+1 {{attribute takes no arguments}}
__attribute__((sycl_global_var(42))) int GlobalWithAttributeArg;

// expected-warning@+1 {{attribute only applies to global variables}}
__attribute__((sycl_global_var)) void F() {
    // expected-warning@+1 {{attribute only applies to global variables}}
    __attribute__((sycl_global_var)) int Local;

    cl::sycl::kernel_single_task<class kernel_name>([=] () {
        (void)GlobalWithAttribute;
        (void)GlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@Inputs/sycl.hpp:* {{called by}}
    });
}