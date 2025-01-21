; Make sure we can turn off asan-stack on host target without the interfering of spirv related flag
; RUN: opt < %s -passes=asan -asan-stack=0 -asan-spir-privates=0 -S | FileCheck %s --check-prefix=CHECK-NOSTACK
; RUN: opt < %s -passes=asan -asan-stack=0 -asan-spir-privates=1 -S | FileCheck %s --check-prefix=CHECK-NOSTACK
; RUN: opt < %s -passes=asan -asan-stack=1 -asan-spir-privates=0 -S | FileCheck %s --check-prefix=CHECK-STACK
; RUN: opt < %s -passes=asan -asan-stack=1 -asan-spir-privates=1 -S | FileCheck %s --check-prefix=CHECK-STACK

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone sanitize_address uwtable
define dso_local i32 @main(i32 noundef %argc) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %a = alloca [10 x i32], align 16
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.lifetime.start.p0(i64 40, ptr %a) #2
  %0 = load i32, ptr %argc.addr, align 4
  %add = add nsw i32 %0, 1
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds [10 x i32], ptr %a, i64 0, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  call void @llvm.lifetime.end.p0(i64 40, ptr %a) #2
  ret i32 %1
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { noinline nounwind optnone sanitize_address uwtable "approx-func-fp-math"="true" "frame-pointer"="all" "loopopt-pipeline"="light" "min-legal-vector-width"="0" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}

; CHECK-STACK: call i64 @__asan_stack_malloc_1(i64 128)
; CHECK-NOSTACK-NOT:  call i64 @__asan_stack_malloc_1(i64 128)
