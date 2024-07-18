// Target "host-x86_64-pc-windows-msvc" only works on Windows
// REQUIRES: system-windows

// Ensure that bundled BC files in archives can work with:
// TEST1:  clang-offload-bundler -list
// TEST2:  clang-offload-bundler -check-section
// TEST3:  clang-offload-bundler -unbundle             with single target
// TEST4:  clang-offload-bundler -unbundle             with multiple targets
//
// In all these tests also ensure functionality with bundled object files still
// works correctly.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make bundled object with targets:
// sycl-spir64-unknown-unknown
// host-x86_64-pc-windows-msvc
// RUN: %clangxx -fsycl --no-offload-new-driver -c %s -o %t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make three distinct BC files
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-device-only -DTYPE1 %s -o %t1.bc
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-device-only -DTYPE2 %s -o %t2.bc
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-device-only -DTYPE3 %s -o %t3.bc

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bundle BC files to different targets:
// host-spir64-unknown-unknown
// host-spir64_gen
// host-spir64_x86_64
// RUN: clang-offload-bundler -type=bc -targets=host-spir64-unknown-unknown -input=%t1.bc  -output=%t1_bundled.bc
// RUN: clang-offload-bundler -type=bc -targets=host-spir64_gen             -input=%t2.bc  -output=%t2_bundled.bc
// RUN: clang-offload-bundler -type=bc -targets=host-spir64_x86_64          -input=%t3.bc  -output=%t3_bundled.bc

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Make archive with bundled BC and o files
// RUN: rm -f %t_bundled.a
// RUN: llvm-ar cr %t_bundled.a %t1_bundled.bc %t2_bundled.bc %t3_bundled.bc %t_bundled.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TEST1
// Check that -list with various archive types can find all targets
// RUN: clang-offload-bundler -list -type=ao  -input=%t_bundled.a > %t_list_ao.txt
// RUN: clang-offload-bundler -list -type=aoo -input=%t_bundled.a > %t_list_aoo.txt
// RUN: FileCheck --check-prefixes=CHECK-LIST < %t_list_ao.txt %s
// RUN: FileCheck --check-prefixes=CHECK-LIST < %t_list_aoo.txt %s

// CHECK-LIST-DAG: sycl-spir64-unknown-unknown
// CHECK-LIST-DAG: host-x86_64-pc-windows-msvc
// CHECK-LIST-DAG: host-spir64-unknown-unknown
// CHECK-LIST-DAG: host-spir64_gen
// CHECK-LIST-DAG: host-spir64_x86_64

// RUN: wc -l %t_list_ao.txt  | FileCheck --check-prefixes=CHECK-LIST-LENGTH %s
// RUN: wc -l %t_list_aoo.txt | FileCheck --check-prefixes=CHECK-LIST-LENGTH %s

// CHECK-LIST-LENGTH: 5

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TEST2
// Test -check-section
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-x86_64-pc-windows-msvc
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_gen
// RUN:     clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_x86_64
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown-a
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-x86_64-pc-windows-msvc-b
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64-unknown-unknown-c
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_gen-d
// RUN: not clang-offload-bundler -check-section -type=ao  -input=%t_bundled.a -targets=host-spir64_x86_64-e
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-x86_64-pc-windows-msvc
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_gen
// RUN:     clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown-a
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-x86_64-pc-windows-msvc-b
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown-c
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_gen-d
// RUN: not clang-offload-bundler -check-section -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64-e

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Unbundle object file to use as a reference result
// RUN: clang-offload-bundler -unbundle -type=o -input=%t_bundled.o -targets=sycl-spir64-unknown-unknown   -output=%t_unbundled_A.o
// RUN: clang-offload-bundler -unbundle -type=o -input=%t_bundled.o -targets=host-x86_64-pc-windows-msvc   -output=%t_unbundled_B.o

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TEST3
// Test archive unbundling
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown   -output=%t_list1.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-x86_64-pc-windows-msvc   -output=%t_list2.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64-unknown-unknown   -output=%t_list3.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64_gen               -output=%t_list4.txt
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=host-spir64_x86_64            -output=%t_list5.txt

// Backtick not supported on Windows
// RUN_disabled_because_backtick_not_supported: cmp %t_unbundled_A.o `cat %t_list1.txt`
// RUN_disabled_because_backtick_not_supported: cmp %t_unbundled_B.o `cat %t_list2.txt`
// RUN_disabled_because_backtick_not_supported: cmp %t1.bc           `cat %t_list3.txt`
// RUN_disabled_because_backtick_not_supported: cmp %t2.bc           `cat %t_list4.txt`
// RUN_disabled_because_backtick_not_supported: cmp %t3.bc           `cat %t_list5.txt`

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TEST4
// Test archive unbundling for multiple targets
// RUN: clang-offload-bundler -unbundle -type=aoo -input=%t_bundled.a -targets=sycl-spir64-unknown-unknown,host-spir64_gen -output=%t_listA.txt -output=%t_listB.txt
// Backtick not supported on Windows
// RUN_disabled_because_backtick_not_supported: cmp %t_unbundled_A.o `cat %t_listA.txt`
// RUN_disabled_because_backtick_not_supported: cmp %t2.bc           `cat %t_listB.txt`

#include <sycl/sycl.hpp>

SYCL_EXTERNAL int foo(int x) {

#ifdef TYPE1
  return x+13;
#elif TYPE2
  return x+17;
#elif TYPE3
  return x+23;
#else
  return x+29;
#endif
}
