// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify %s

#include "Inputs/sycl.hpp"

// expected-error@+2 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]][[intel::named_sub_group_size(automatic)]]
void f1();
// expected-error@+2 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]][[intel::sub_group_size(1)]]
void f2();

// expected-error@+1 {{'sub_group_size' and 'named_sub_group_size' attributes are not compatible}}
[[intel::sub_group_size(1)]]
void f3();
// expected-note@+1 {{conflicting attribute is here}}
[[intel::named_sub_group_size(primary)]]
void f3();

// expected-error@+1 {{'named_sub_group_size' and 'sub_group_size' attributes are not compatible}}
[[intel::named_sub_group_size(primary)]]
void f4();
// expected-note@+1 {{conflicting attribute is here}}
[[intel::sub_group_size(1)]]
void f4();

// expected-note@+1 {{previous attribute is here}}
[[intel::named_sub_group_size(automatic)]]
void f5();

// expected-warning@+1 {{attribute 'named_sub_group_size' is already applied with different arguments}}
[[intel::named_sub_group_size(primary)]]
void f5();

[[intel::named_sub_group_size(automatic)]]
void f6();

[[intel::named_sub_group_size(automatic)]]
void f6();

// expected-warning@+1 {{'named_sub_group_size' attribute argument not supported: 'invalid'}}
[[intel::named_sub_group_size(invalid)]]
void f7();
