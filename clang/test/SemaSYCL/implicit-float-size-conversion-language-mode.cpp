// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify=size \
// RUN:   -Wimplicit-float-size-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify=size \
// RUN:   -fsycl-is-host -Wimplicit-float-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify=size \
// RUN:   -fsycl-is-host -Wsycl-implicit-float-size-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify=size \
// RUN:   -Wimplicit-float-size-conversion -Wno-implicit-float-conversion %s

// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify \
// RUN:   -Wimplicit-float-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify \
// RUN:   -Wsycl-implicit-float-size-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify \
// RUN:   -fsycl-is-host -Wimplicit-float-conversion -Wno-implicit-float-size-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify \
// RUN:   -fsycl-is-host -Wimplicit-float-conversion -Wno-sycl-implicit-float-size-conversion %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -verify \
// RUN:   -fsycl-is-host -Wimplicit-float-size-conversion -Wno-implicit-float-conversion %s

// This test checks that -Wimplicit-float-conversion and -Wimplicit-float-size-conversion behave
// correctly:
// -W(no-)implicit-float-conversion implies -W(no-)implicit-float-size-conversion for SYCL only,
// but -W(no-)implicit-float-size-conversion can be used outside SYCL too

// expected-no-diagnostics
float s = 1.0; // size-warning {{implicit conversion between floating point types of different sizes}}
