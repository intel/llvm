; Make sure we can turn off asan-stack on host target without the interfering of spirv related flag
; RUN: opt < %s -passes=asan -asan-stack=0 -asan-spir-privates=0 -S | FileCheck %s --prefix=CHECK-STACK
; RUN: opt < %s -passes=asan -asan-stack=0 -asan-spir-privates=1 -S | FileCheck %s --prefix=CHECK-STACK
; RUN: opt < %s -passes=asan -asan-stack=1 -asan-spir-privates=0 -S | FileCheck %s --prefix=CHECK-NOSTACK
; RUN: opt < %s -passes=asan -asan-stack=1 -asan-spir-privates=1 -S | FileCheck %s --prefix=CHECK-NOSTACK

target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @main(i32 noundef %argc) {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %a = alloca [10 x i32], align 16
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  %0 = load i32, ptr %argc.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [10 x i32], ptr %a, i64 0, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

; CHECK-STACK: __asan_stack_malloc_1
; CHECK-NOSTACK-NOT:  __asan_stack_malloc_1
