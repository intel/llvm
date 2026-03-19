// Test that __builtin_num_fields, __builtin_num_bases, __builtin_field_type,
// and __builtin_base_type work correctly with PCH.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++17 -triple spir64-unknown-unknown -fsycl-is-device -emit-pch -x c++-header \
// RUN:   %t/pch.h -o %t/pch.h.device.pch
// RUN: %clang_cc1 -std=c++17 -triple spir64-unknown-unknown -fsycl-is-device -verify \
// RUN:   -include-pch %t/pch.h.device.pch %t/test.cpp

#--- pch.h
struct Base {};
template <typename T> struct X {
  static_assert(__is_same(Base, decltype(__builtin_base_type(T, 0))));
  static_assert(__is_same(int, decltype(__builtin_field_type(T, 0))));
  static_assert(__builtin_num_fields(T) == 1);
  static_assert(__builtin_num_bases(T) == 1);
};

#--- test.cpp
// expected-no-diagnostics
struct A : public Base { int i; };
void foo() {
  X<A> x;
}

