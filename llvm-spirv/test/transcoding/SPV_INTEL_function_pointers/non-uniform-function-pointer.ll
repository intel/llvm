; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM
;
; Generated from:
; int foo(int v) {
;   return v + 1;
; }
;
; int bar(int v) {
;   return v + 2;
; }
;
; __kernel void test(__global int *data, int control) {
;   int (*fp)(int) = 0;
;
;   if (get_global_id(0) % control == 0)
;     fp = &foo;
;   else
;     fp = &bar;
;
;   data[get_global_id(0)] = fp(data[get_global_id(0)]);
; }
;
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
;
; CHECK-SPIRV: EntryPoint 6 [[KERNEL_ID:[0-9]+]] "test"
; CHECK-SPIRV: TypeInt [[TYPE_INT32_ID:[0-9+]]] 32
; CHECK-SPIRV: TypeFunction [[FOO_TYPE_ID:[0-9]+]] [[TYPE_INT32_ID]] [[TYPE_INT32_ID]]
; CHECK-SPIRV: TypePointer [[FOO_PTR_TYPE_ID:[0-9]+]] {{[0-9]+}} [[FOO_TYPE_ID]]
; CHECK-SPIRV: TypePointer [[FOO_PTR_ALLOCA_TYPE_ID:[0-9]+]] 7 [[FOO_PTR_TYPE_ID]]
; CHECK-SPIRV: ConstFunctionPointerINTEL [[FOO_PTR_TYPE_ID]] [[FOO_PTR_ID:[0-9]+]] [[FOO_ID:[0-9]+]]
; CHECK-SPIRV: ConstFunctionPointerINTEL [[FOO_PTR_TYPE_ID]] [[BAR_PTR_ID:[0-9]+]] [[BAR_ID:[0-9]+]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[FOO_ID]] {{[0-9]+}} [[FOO_TYPE_ID]]
; CHECK-SPIRV: Function {{[0-9]+}} [[BAR_ID]] {{[0-9]+}} [[FOO_TYPE_ID]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[KERNEL_ID]]
; CHECK-SPIRV: Variable [[FOO_PTR_ALLOCA_TYPE_ID]] [[FOO_PTR_ALLOCA_ID:[0-9]+]]
; CHECK-SPIRV: Store [[FOO_PTR_ALLOCA_ID]] [[FOO_PTR_ID]]
; CHECK-SPIRV: Store [[FOO_PTR_ALLOCA_ID]] [[BAR_PTR_ID]]
; CHECK-SPIRV: Load [[FOO_PTR_TYPE_ID]] [[LOADED_FOO_PTR:[0-9]+]] [[FOO_PTR_ALLOCA_ID]]
; CHECK-SPIRV: FunctionPointerCallINTEL {{[0-9]+}} {{[0-9]+}} [[LOADED_FOO_PTR]]
;
; CHECK-LLVM: define spir_kernel void @test
; CHECK-LLVM: %fp = alloca i32 (i32)*
; CHECK-LLVM: store i32 (i32)* @foo, i32 (i32)** %fp
; CHECK-LLVM: store i32 (i32)* @bar, i32 (i32)** %fp
; CHECK-LLVM: %[[FP:.*]] = load i32 (i32)*, i32 (i32)** %fp
; CHECK-LLVM: call spir_func i32 %[[FP]](i32 %{{.*}})


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @foo(i32 %v) #0 {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, i32* %v.addr, align 4
  %0 = load i32, i32* %v.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @bar(i32 %v) #0 {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, i32* %v.addr, align 4
  %0 = load i32, i32* %v.addr, align 4
  %add = add nsw i32 %0, 2
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test(i32 addrspace(1)* %data, i32 %control) #1 !kernel_arg_addr_space !1 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %data.addr = alloca i32 addrspace(1)*, align 8
  %control.addr = alloca i32, align 4
  %fp = alloca i32 (i32)*, align 8
  store i32 addrspace(1)* %data, i32 addrspace(1)** %data.addr, align 8
  store i32 %control, i32* %control.addr, align 4
  store i32 (i32)* null, i32 (i32)** %fp, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %0 = load i32, i32* %control.addr, align 4
  %conv = sext i32 %0 to i64
  %rem = urem i64 %call, %conv
  %cmp = icmp eq i64 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 (i32)* @foo, i32 (i32)** %fp, align 8
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 (i32)* @bar, i32 (i32)** %fp, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %1 = load i32 (i32)*, i32 (i32)** %fp, align 8
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %data.addr, align 8
  %call2 = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 %call2
  %3 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call3 = call spir_func i32 %1(i32 %3) #4
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %data.addr, align 8
  %call4 = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %call4
  store i32 %call3, i32 addrspace(1)* %arrayidx5, align 4
  ret void
}

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #2

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind readnone }
attributes #4 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 7.1.0 "}
!4 = !{!"none", !"none"}
!5 = !{!"int*", !"int"}
!6 = !{!"", !""}
