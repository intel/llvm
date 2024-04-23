// RUN: %clang_cc1 -verify %s

kernel __attribute__((work_group_size_hint(8,16,32,4))) void kernel6() {} //expected-error{{'work_group_size_hint' attribute requires exactly 3 arguments}}

kernel __attribute__((work_group_size_hint(1,2))) void kernel6b() {}  //expected-error{{'work_group_size_hint' attribute requires exactly 3 arguments}}

kernel __attribute__((reqd_work_group_size(8,16,32,4))) void kernel6c() {} //expected-error{{'reqd_work_group_size' attribute requires exactly 3 arguments}}

kernel __attribute__((reqd_work_group_size(1,2))) void kernel6d() {}  //expected-error{{'reqd_work_group_size' attribute requires exactly 3 arguments}}

kernel __attribute__((work_group_size_hint(1,2,3))) __attribute__((work_group_size_hint(3,2,1))) void kernel7() {}  //expected-warning{{attribute 'work_group_size_hint' is already applied with different arguments}}

__attribute__((reqd_work_group_size(8,16,32))) void kernel8(){} // expected-error {{attribute 'reqd_work_group_size' can only be applied to an OpenCL kernel}}

__attribute__((work_group_size_hint(8,16,32))) void kernel9(){} // expected-error {{attribute 'work_group_size_hint' can only be applied to an OpenCL kernel}}

constant int foo1 __attribute__((reqd_work_group_size(8,16,32))) = 0; // expected-error {{'reqd_work_group_size' attribute only applies to functions}}

constant int foo2 __attribute__((work_group_size_hint(8,16,32))) = 0; // expected-error {{'work_group_size_hint' attribute only applies to functions}}

void f_kernel_image2d_t( kernel image2d_t image ) { // expected-error {{'kernel' attribute only applies to functions}}
  int __kernel x; // expected-error {{'__kernel' attribute only applies to functions}}
}

kernel __attribute__((reqd_work_group_size(1,2,0))) void kernel11(){} // expected-error {{'reqd_work_group_size' attribute must be greater than 0}}
kernel __attribute__((reqd_work_group_size(1,0,2))) void kernel12(){} // expected-error {{'reqd_work_group_size' attribute must be greater than 0}}
kernel __attribute__((reqd_work_group_size(0,1,2))) void kernel13(){} // expected-error {{'reqd_work_group_size' attribute must be greater than 0}}

__kernel __attribute__((work_group_size_hint(8,-16,32))) void neg1() {} //expected-error{{'work_group_size_hint' attribute requires a non-negative integral compile time constant expression}}
__kernel __attribute__((reqd_work_group_size(8,16,-32))) void neg2(){} // expected-error{{'reqd_work_group_size' attribute requires a non-negative integral compile time constant expression}}

// 4294967294 is a negative integer if treated as signed.
// Should compile successfully, since we expect an unsigned.
__kernel __attribute__((reqd_work_group_size(8,16,4294967294))) void ok1(){}
