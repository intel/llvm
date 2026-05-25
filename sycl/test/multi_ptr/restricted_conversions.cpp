#include <sycl/detail/core.hpp>
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note

template <typename T, sycl::access::address_space Space,
          sycl::access::decorated Decorated>
using multi_ptr_t = sycl::multi_ptr<T, Space, Decorated>;

using legacy_ptr = multi_ptr_t<int, sycl::access::address_space::private_space,
                               sycl::access::decorated::legacy>;
using non_legacy_ptr =
    multi_ptr_t<int, sycl::access::address_space::private_space,
                sycl::access::decorated::no>;
using private_ptr = multi_ptr_t<int, sycl::access::address_space::private_space,
                                sycl::access::decorated::no>;
using const_void_ptr =
    multi_ptr_t<const void, sycl::access::address_space::private_space,
                sycl::access::decorated::no>;
using void_ptr = multi_ptr_t<void, sycl::access::address_space::private_space,
                             sycl::access::decorated::no>;
using global_ptr = multi_ptr_t<int, sycl::access::address_space::global_space,
                               sycl::access::decorated::no>;
using local_ptr = multi_ptr_t<int, sycl::access::address_space::local_space,
                              sycl::access::decorated::no>;
// expected-warning@+2 {{constant_space' is deprecated: sycl::access::address_space::constant_space is deprecated since SYCL 2020}}
using constant_ptr =
    multi_ptr_t<const int, sycl::access::address_space::constant_space,
                sycl::access::decorated::no>;
using generic_ptr =
    multi_ptr_t<const int, sycl::access::address_space::generic_space,
                sycl::access::decorated::no>;

legacy_ptr leg_ptr{nullptr};
non_legacy_ptr nonleg_ptr{nullptr};

// expected-error@+1 {{no matching constructor for initialization of}}
non_legacy_ptr nonleg_ptr1{leg_ptr};
// expected-error@+1 {{no viable conversion from}}
non_legacy_ptr nonleg_ptr2 = leg_ptr;
// TODO: is constructor legal?
// expected-warning@+1 {{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead}}
legacy_ptr leg_ptr1{nonleg_ptr};
// expected-error@+1 {{no viable conversion from 'multi_ptr_t<}}
legacy_ptr leg_ptr2 = nonleg_ptr;

const_void_ptr const_void{nullptr};

// expected-error@+1 {{no matching constructor for initialization of 'void_ptr'}}
void_ptr void_ptr1{const_void};
// expected-error@+1 {{no viable conversion from 'multi_ptr_t<}}
void_ptr void_ptr2 = const_void;

global_ptr global{nullptr};

private_ptr private_ptr_instance{nullptr};

// expected-error@+1 {{no matching constructor for initialization of 'local_ptr'}}
local_ptr local{global};

// expected-error@+1 {{no viable conversion from 'multi_ptr_t<}}
local_ptr local1 = global;

// expected-error@+1 {{no matching constructor for initialization of 'global_ptr'}}
global_ptr global1{local};

// expected-error@+1 {{no viable conversion from 'multi_ptr_t<}}
global_ptr global2 = local;

// expected-error@+1 {{no matching constructor for initialization of 'local_ptr'}}
local_ptr local_from_private{private_ptr_instance};

// expected-error@+1 {{no viable conversion from 'multi_ptr_t<}}
local_ptr local_from_private_implicit = private_ptr_instance;

// expected-error@+1 {{no matching constructor for initialization of 'global_ptr'}}
global_ptr global_from_private{private_ptr_instance};

// expected-error@+1 {{no matching constructor for initialization of 'private_ptr'}}
private_ptr private_from_local{local};

// expected-error@+1 {{no matching constructor for initialization of 'private_ptr'}}
private_ptr private_from_global{global};

// expected-warning@+1 2 {{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
bool private_equals_local = private_ptr_instance == local;

// expected-warning@+1 2 {{'operator int *' is deprecated: Conversion to pointer type is deprecated since SYCL 2020. Please use get() instead.}}
bool private_less_than_local = private_ptr_instance < local;

