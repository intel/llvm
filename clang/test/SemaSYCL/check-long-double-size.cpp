// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -aux-triple x86_64-unknown-linux-gnu -fsyntax-only -DLONG_DOUBLE_SIZE=16 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir-unknown-unknown-sycldevice -aux-triple i386-pc-linux-gnu -fsyntax-only -DLONG_DOUBLE_SIZE=12 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows-sycldevice -aux-triple x86_64-pc-windows-msvc -fsyntax-only -DLONG_DOUBLE_SIZE=8 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir-unknown-windows-sycldevice -aux-triple i686-pc-windows-msvc -fsyntax-only -DLONG_DOUBLE_SIZE=8 -verify %s
// expected-no-diagnostics

static_assert(sizeof(long double) == LONG_DOUBLE_SIZE, "wrong sizeof long double");
