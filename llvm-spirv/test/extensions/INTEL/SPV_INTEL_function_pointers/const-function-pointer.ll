; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV: Name [[F1Name:[0-9]+]] "f1"
; CHECK-SPIRV: Name [[F2Name:[0-9]+]] "f2"
; CHECK-SPIRV-DAG: TypeInt [[Int32:[0-9]+]] 32
; CHECK-SPIRV-DAG: TypeInt [[Int64:[0-9]+]] 64
; CHECK-SPIRV-DAG: Constant [[Int32]] [[XArg:[0-9]+]] 32
; CHECK-SPIRV-DAG: Constant [[Int32]] [[YArg:[0-9]+]] 2

; CHECK-SPIRV: ConstantFunctionPointerINTEL {{[0-9]+}} [[F1:[0-9]+]] [[F1Name]]
; CHECK-SPIRV: ConstantFunctionPointerINTEL {{[0-9]+}} [[F2:[0-9]+]] [[F2Name]]
; CHECK-SPIRV: ConstantComposite {{[0-9]+}} [[ConstComp:[0-9]+]] [[F1]] [[F2]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[Var:[0-9]+]] {{[0-9]+}} [[ConstComp]]

; CHECK-SPIRV: InBoundsPtrAccessChain {{[0-9]+}} [[GEP:[0-9]+]] [[Var]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: Load {{[0-9]+}} [[FuncPtr:[0-9]+]] [[GEP]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[Int32]] {{[0-9]+}} [[FuncPtr]] [[XArg]] [[YArg]]

; CHECK-LLVM: @__const.main.funcs = internal addrspace(1) constant [2 x ptr] [ptr @f1, ptr @f2], align 16
; CHECK-LLVM:   %[[Idx:[a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr addrspace(1) @__const.main.funcs, i64 0, i64 %{{[a-z0-9]+}}
; CHECK-LLVM:   %[[FuncPtr:[a-z0-9]+]] = load ptr, ptr addrspace(1) %[[Idx]], align 8
; CHECK-LLVM:   %{{[a-z0-9]+}} = call spir_func i32 %[[FuncPtr]](i32 32, i32 2)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

@__const.main.funcs = private unnamed_addr addrspace(1) constant [2 x ptr] [ptr @f1, ptr @f2], align 16

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @f1(i32 %a, i32 %b) #0 {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @f2(i32 %a, i32 %b) #0 {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #1 {
entry:
  %call = tail call i32 @rand() #3
  %rem = srem i32 %call, 2
  %idxprom = sext i32 %rem to i64
  %arrayidx = getelementptr inbounds [2 x ptr], ptr addrspace(1) @__const.main.funcs, i64 0, i64 %idxprom
  %0 = load ptr, ptr addrspace(1) %arrayidx, align 8
  %call1 = tail call i32 %0(i32 32, i32 2) #3
  ret i32 %call1
}

; Function Attrs: nounwind
declare dso_local i32 @rand() local_unnamed_addr #2

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
