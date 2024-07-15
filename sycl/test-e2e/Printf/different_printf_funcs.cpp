// This test is written with an aim to make sure that `printf`,
// `__builtin_printf` and `experimenta::printf` are handled correctly and
// resolved to a call to `vprintf` (which is what CUDA expects), while
// generating correct output.
//
// UNSUPPORTED: hip_amd
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// RUN: %{build} -fsycl-device-only -S -emit-llvm -o - | FileCheck --check-prefix=CHECK-IR %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include "helper.hpp"

using namespace sycl;

// CHECK-IR: @.str = private unnamed_addr addrspace(1) constant [35 x i8] c"Printf - literal strings: s=%s %s\0A\00", align 1
// CHECK-IR: @.str1 = private unnamed_addr addrspace(1) constant [6 x i8] c"Hello\00", align 1
// CHECK-IR: @.str2 = private unnamed_addr addrspace(1) constant [7 x i8] c"World!\00", align 1
// CHECK-IR: @.str3 = private unnamed_addr addrspace(1) constant [43 x i8] c"Builtin printf - literal strings: s=%s %s\0A\00", align 1
// CHECK-IR: @.str4 = private unnamed_addr addrspace(1) constant [48 x i8] c"Experimental printf - literal strings: s=%s %s\0A\00", align 1

 
// CHECK-IR: call i32 @vprintf(ptr addrspacecast (ptr addrspace(1) @.str to ptr)
void do_printf_test() {
  printf("Printf - literal strings: s=%s %s\n", "Hello",
         "\x57\x6F\x72\x6C\x64\x21");
}

// CHECK-IR: call i32 @vprintf(ptr addrspacecast (ptr addrspace(1) @.str3 to ptr)
void do_builtin_printf_test() {
  __builtin_printf("Builtin printf - literal strings: s=%s %s\n", "Hello",
                   "\x57\x6F\x72\x6C\x64\x21");
}

// CHECK-IR: call noundef i32 @vprintf(ptr addrspacecast (ptr addrspace(1) @.str4 to ptr)
void do_experimental_printf_test() {
  ext::oneapi::experimental::printf(
      "Experimental printf - literal strings: s=%s %s\n", "Hello",
      "\x57\x6F\x72\x6C\x64\x21");
}
class PrintfTester;

int main() {
  queue q;

  q.submit([](handler &cgh) {
    cgh.single_task<PrintfTester>([]() {
      // CHECK: Printf - literal strings: s=Hello World!
      do_printf_test();
      // CHECK-NEXT: Builtin printf - literal strings: s=Hello World!
      do_builtin_printf_test();
      // CHECK-NEXT: Experimental printf - literal strings: s=Hello World!
      do_experimental_printf_test();
    });
  });
  q.wait();

  return 0;
}
