// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl-is-device -S -emit-llvm %s -o - | FileCheck %s

// This test checks the generation of CMGenxSIMT function attributes

__attribute__((sycl_device)) __attribute__((sycl_esimd_widen(32))) void foo1() {}
// CHECK: @_Z4foo1v() #[[ATTR1:[0-9]+]]

__attribute__((sycl_device)) __attribute__((sycl_esimd_widen(8))) __attribute__((sycl_esimd_widen(16))) void foo2() {}
// CHECK: @_Z4foo2v() #[[ATTR2:[0-9]+]]

// CHECK: attributes #[[ATTR1]] = { {{.*}} "CMGenxSIMT"="32" {{.*}}}
// CHECK: attributes #[[ATTR2]] = { {{.*}} "CMGenxSIMT"="8" {{.*}}}
