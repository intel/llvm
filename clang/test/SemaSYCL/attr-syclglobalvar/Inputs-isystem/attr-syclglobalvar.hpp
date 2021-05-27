#include "../../Inputs/sycl.hpp"

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
        (void)HppStaticGlobal.InstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)StaticLocalVar; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)HppAnonymousStaticUnionInstanceMember; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
        (void)HppGlobalNoAttribute; // expected-error {{SYCL kernel cannot use a non-const global variable}} expected-note@../../Inputs/sycl.hpp:* {{called by}}
    });
}
