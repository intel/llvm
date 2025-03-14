; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_variable_length_array
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; The IR was generated for the following C code:
; int foo(long a, long b) {
;     int qqq[42];
;     int t = 0;
;     {
;       int arr[a];
;       t = arr[b];
;     }
;     int brr[a];
;     return t + qqq[b] + brr[b];
; }
; Command: clang example.c -emit-llvm -O1 -g0 -fno-discard-value-names
; Target triple and datalayout was changed manually.

; CHECK-SPIRV: Capability VariableLengthArrayINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_variable_length_array"

; CHECK-SPIRV-DAG: TypeInt [[#Int:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Char:]] 8 0
; CHECK-SPIRV-DAG: TypePointer [[#CharPtr:]] [[#]] [[#Char]]
; CHECK-SPIRV-DAG: TypePointer [[#IntPtr:]] [[#]] [[#Int]]

; CHECK-SPIRV: FunctionParameter [[#]] [[#A:]]

; CHECK-SPIRV: SaveMemoryINTEL [[#CharPtr]] [[#SavedMem:]]
; CHECK-SPIRV: VariableLengthArrayINTEL [[#IntPtr]] [[#]] [[#A]]
; CHECK-SPIRV: RestoreMemoryINTEL [[#SavedMem]]

; CHECK-SPIRV: SaveMemoryINTEL [[#CharPtr]] [[#SavedMem:]]
; CHECK-SPIRV: VariableLengthArrayINTEL [[#IntPtr]] [[#]] [[#A]]
; CHECK-SPIRV: RestoreMemoryINTEL [[#SavedMem]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

define dso_local spir_func i32 @foo(i64 %a, i64 %b) local_unnamed_addr #0 {
entry:
  %qqq = alloca [42 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 168, ptr nonnull %qqq) #2

; CHECK-LLVM: %[[#SavedMem:]] = call ptr @llvm.stacksave.p0()
  %0 = call ptr @llvm.stacksave.p0()

; CHECK-LLVM: alloca i32, i64 %a, align 16
  %vla = alloca i32, i64 %a, align 16

  %arrayidx = getelementptr inbounds i32, ptr %vla, i64 %b
  %1 = load i32, ptr %arrayidx, align 4

; CHECK-LLVM: call void @llvm.stackrestore.p0(ptr %[[#SavedMem]])
  call void @llvm.stackrestore.p0(ptr %0)

; CHECK-LLVM: %[[#SavedMem:]] = call ptr @llvm.stacksave.p0()
  %2 = call ptr @llvm.stacksave.p0()

; CHECK-LLVM: alloca i32, i64 %a, align 16
  %vla2 = alloca i32, i64 %a, align 16

  %arrayidx3 = getelementptr inbounds [42 x i32], ptr %qqq, i64 0, i64 %b
  %3 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %3, %1
  %arrayidx4 = getelementptr inbounds i32, ptr %vla2, i64 %b
  %4 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %add, %4

; CHECK-LLVM: call void @llvm.stackrestore.p0(ptr %[[#SavedMem]])
  call void @llvm.stackrestore.p0(ptr %2)

  call void @llvm.lifetime.end.p0(i64 168, ptr nonnull %qqq) #2
  ret i32 %add5
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

declare ptr @llvm.stacksave.p0() #2

declare void @llvm.stackrestore.p0(ptr) #2

declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5804a8b1228ba890d48f4085a3a192ef83c73e00)"}
