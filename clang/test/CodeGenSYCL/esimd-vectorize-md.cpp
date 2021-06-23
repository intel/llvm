// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice \
// RUN:   -fsycl-is-device -S -emit-llvm %s -o - | FileCheck %s

// This test checks the generation of CMGenxSIMT function attributes

[[intel::sycl_esimd_vectorize(32)]] __attribute__((sycl_device)) void foo1() {}
// CHECK: @_Z4foo1v() #[[ATTR1:[0-9]+]]

[[intel::sycl_esimd_vectorize(8)]] [[intel::sycl_esimd_vectorize(16)]] __attribute__((sycl_device)) void foo2() {}
// CHECK: @_Z4foo2v() #[[ATTR2:[0-9]+]]

[[intel::sycl_esimd_vectorize(8)]] __attribute__((sycl_device)) void foo3();
[[intel::sycl_esimd_vectorize(16)]] __attribute__((sycl_device)) void foo3() {}
// CHECK: @_Z4foo3v() #[[ATTR3:[0-9]+]]

// CHECK: attributes #[[ATTR1]] = { {{.*}} "CMGenxSIMT"="32" {{.*}}}
// CHECK: attributes #[[ATTR2]] = { {{.*}} "CMGenxSIMT"="8" {{.*}}}
// CHECK: attributes #[[ATTR3]] = { {{.*}} "CMGenxSIMT"="16" {{.*}}}
