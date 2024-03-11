; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM
;
; Generated from:
; __attribute__((referenced_indirectly))
; int foo(int arg) {
;   return arg + 10;
; }
;
; void __kernel test(__global int *data, int input) {
;   int (__constant *fp)(int) = &foo;
;
;   *data = fp(input);
; }
;
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Capability IndirectReferencesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
;
; CHECK-SPIRV: Name [[FOO_ID:[0-9]+]] "foo"
; CHECK-SPIRV: Decorate [[FOO_ID]] ReferencedIndirectlyINTEL
; CHECK-SPIRV: Function {{[0-9]+}} [[FOO_ID]]
;
; CHECK-LLVM: define spir_func i32 @foo(i32 %arg) #[[ATTRS:[0-9]+]]
; CHECK-LLVM: attributes #[[ATTRS]] = {{.*}} "referenced-indirectly"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @foo(i32 %arg) #0 {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, ptr %arg.addr, align 4
  %0 = load i32, ptr %arg.addr, align 4
  %add = add nsw i32 %0, 10
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test(ptr addrspace(1) %data, i32 %input) #1 !kernel_arg_addr_space !1 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %data.addr = alloca ptr addrspace(1), align 8
  %input.addr = alloca i32, align 4
  %fp = alloca ptr, align 8
  store ptr addrspace(1) %data, ptr %data.addr, align 8
  store i32 %input, ptr %input.addr, align 4
  store ptr @foo, ptr %fp, align 8
  %0 = load ptr, ptr %fp, align 8
  %1 = load i32, ptr %input.addr, align 4
  %call = call spir_func i32 %0(i32 %1) #2
  %2 = load ptr addrspace(1), ptr %data.addr, align 8
  store i32 %call, ptr addrspace(1) %2, align 4
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" "referenced-indirectly" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{!"clang version 7.0.0 "}
!5 = !{!"none", !"none"}
!6 = !{!"int*", !"int"}
!7 = !{!"", !""}

