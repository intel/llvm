; RUN: %if hip_amd %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple amdgcn-amd-amdhsa -passes=sycl-spec-const-materializer -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER %s %}

; RUN: %if cuda %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple nvptx64-nvidia-cuda -passes=sycl-spec-const-materializer -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER %s %}

; RUN: %if hip_amd %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple amdgcn-amd-amdhsa -passes=sycl-spec-const-materializer,early-cse,adce -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER-CSE %s %}

; RUN: %if cuda %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple nvptx64-nvidia-cuda -passes=sycl-spec-const-materializer,early-cse,adce -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER-CSE %s %}

source_filename = "basic.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; Check the basic replacement of specialization constant. We expect 2 global ;
; variables (i32 and [2 x i32]), no loads from implicit kernel argument:
; CHECK-MATERIALIZER.

; For CHECK-MATERIALIZER-CSE also include early commons subexpression
; elimination pass and expect the final literal to be stored to the output
; pointer.

;CHECK-MATERIALIZER: @SpecConsBlob___test_kernel_0 = weak_odr addrspace(4) constant i32 7
;CHECK-MATERIALIZER: @SpecConsBlob___test_kernel_1 = weak_odr addrspace(4) constant [2 x i32] [i32 3, i32 1]


;CHECK: __test_kernel
define weak_odr protected amdgpu_kernel void @__test_kernel(ptr addrspace(1) noundef align 4 %out, ptr addrspace(1) noundef align 1 %_arg__specialization_constants_buffer) {
entry:
  ;CHECK-MATERIALIZER-CSE-NOT: addrspacecast ptr addrspace(1) %_arg__specialization_constants_buffer to ptr
  ;CHECK-MATERIALIZER: [[REG1:%[0-9]+]] = load i32, ptr addrspace(4) @SpecConsBlob___test_kernel_0
  ;CHECK-MATERIALIZER: [[REG2:%[0-9]+]] = load [2 x i32], ptr addrspace(4) @SpecConsBlob___test_kernel_1
  %0 = addrspacecast ptr addrspace(1) %_arg__specialization_constants_buffer to ptr
  %gep = getelementptr i8, ptr %0, i32 0
  %bc = bitcast ptr %gep to ptr
  ;CHECK-MATERIALIZER-CSE-NOT: load i32, ptr %bc
  %load1 = load i32, ptr %bc, align 4
  %gep1 = getelementptr i8, ptr %0, i32 4
  %bc2 = bitcast ptr %gep1 to ptr
  ;CHECK-MATERIALIZER-CSE-NOT: load [2 x i32], ptr %bc2
  %load2 = load [2 x i32], ptr %bc2, align 4
  ;CHECK-MATERIALIZER: load i32, ptr addrspace(4) @SpecConsBlob___test_kernel_0
  %straight_load = load i32, ptr %0, align 4
  %extract1 = extractvalue [2 x i32] %load2, 0
  %extract2 = extractvalue [2 x i32] %load2, 1
  %add1 = add nsw i32 %extract1, %load1
  %add2 = add nsw i32 %add1, %extract2
  %add3 = add nsw i32 %add2, %straight_load
  ;CHECK-MATERIALIZER-CSE: store i32 18, ptr addrspace(1) %out,
  ;CHECK-MATERIALIZER: %extract1 = extractvalue [2 x i32] [[REG2]], 0
  ;CHECK-MATERIALIZER: %extract2 = extractvalue [2 x i32] [[REG2]], 1
  ;CHECK-MATERIALIZER: %add1 = add nsw i32 %extract1, [[REG1]]

  store i32 %add3, ptr addrspace(1) %out, align 4
  ret void
}

!SYCL_SpecConst_data = !{!1}
!1 = !{!"\07\00\00\00\03\00\00\00\01\00\00\00"}
