; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-INTRINSIC
; RUN: not llvm-spirv %t.bc -spirv-allow-unknown-intrinsics -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ALLOCA

; CHECK-INTRINSIC: InvalidFunctionCall: Unexpected llvm intrinsic:
; CHECK-INTRINSIC-NEXT: call i8* @llvm.stacksave()
; CHECK-INTRINSIC-NEXT: Translation of llvm.stacksave intrinsic requires SPV_INTEL_variable_length_array extension or -spirv-allow-unknown-intrinsics option.

; CHECK-ALLOCA: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ALLOCA-NEXT: %vla = alloca i32, i64 %a
; CHECK-ALLOCA-NEXT: Translation of dynamic alloca requires SPV_INTEL_variable_length_array extension.

; The IR was generated for the following C code:
; int bar();
; int foo(long a, long b) {
;     int t = 0;
;     {
;       int arr[a];
;       t = arr[b];
;     }
;     return t + bar();
; }

; Command: clang example.c -emit-llvm -O1 -g0 -fno-discard-value-names
; Target triple and datalayout was changed manually.

source_filename = "./example.c"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"


define dso_local i32 @_Z3fooll(i64 %a, i64 %b) local_unnamed_addr #0 {
entry:
  %0 = call i8* @llvm.stacksave()
  %vla = alloca i32, i64 %a, align 16
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 %b
  %1 = load i32, i32* %arrayidx, align 4
  call void @llvm.stackrestore(i8* %0)
  %call = call i32 @_Z3barv()
  %add = add nsw i32 %call, %1
  ret i32 %add
}

declare i8* @llvm.stacksave() #1

declare void @llvm.stackrestore(i8*) #1

declare dso_local i32 @_Z3barv() local_unnamed_addr #2

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

