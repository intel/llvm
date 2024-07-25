; RUN: %if hip_amd %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple amdgcn-amd-amdhsa -passes=sycl-spec-const-materializer -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER %s %}

; RUN: %if cuda %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple nvptx64-nvidia-cuda -passes=sycl-spec-const-materializer -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER %s %}

; RUN: %if hip_amd %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple amdgcn-amd-amdhsa -passes=sycl-spec-const-materializer,early-cse -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER-CSE %s %}

; RUN: %if cuda %{ opt -load-pass-plugin %shlibdir/SYCLKernelJIT%shlibext\
; RUN: --mtriple nvptx64-nvidia-cuda -passes=sycl-spec-const-materializer,early-cse -S %s |\
; RUN: FileCheck --check-prefix=CHECK-MATERIALIZER-CSE %s %}

source_filename = "multi_type.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; The same logic as in the basic.ll, but with more complicated types; an array
; of struct with multiple members. This is important as the pass has to
; correctly set the type of offsetted memory region corresponding to
; specialization constant (see getConstantOfType in the pass).
; For CHECK-MATERIALIZER-CSE expect literal store only.

;CHECK-MATERIALIZER: @SpecConsBlob___test_kernel_0 = weak_odr addrspace(4) constant i32
;CHECK-MATERIALIZER: @SpecConsBlob___test_kernel_1 = weak_odr addrspace(4) constant %"struct.std::array"
;CHECK-MATERIALIZER: @SpecConsBlob___test_kernel_2 = weak_odr addrspace(4) constant [2 x i32]

%"struct.std::array" = type { [5 x %struct.ThreePrimitives] }
%struct.ThreePrimitives = type <{ double, i64, half }>

;CHECK: __test_kernel
define weak_odr protected amdgpu_kernel void @__test_kernel(ptr addrspace(1) noundef align 4 %out, ptr addrspace(1) noundef align 1 %_arg__specialization_constants_buffer) {
entry:
  ;CHECK-MATERIALIZER-CSE-NOT: load
  ;CHECK-MATERIALIZER-CSE: store double 1.100000e+01, ptr addrspace(1) %out

  ;CHECK-MATERIALIZER-CSE-NOT: addrspacecast ptr addrspace(1) %_arg__specialization_constants_buffer to ptr
  %0 = addrspacecast ptr addrspace(1) %_arg__specialization_constants_buffer to ptr
  %gep = getelementptr i8, ptr %0, i32 0
  %bc = bitcast ptr %gep to ptr
  ;CHECK-MATERIALIZER: load i32, ptr addrspace(4) @SpecConsBlob___test_kernel_0
  %load = load i32, ptr %bc, align 4

  %gep2 = getelementptr i8, ptr %0, i32 4
  %bc2 = bitcast ptr %gep2 to ptr
  ;CHECK-MATERIALIZER: load [2 x i32], ptr addrspace(4) @SpecConsBlob___test_kernel_2
  %load2 = load [2 x i32], ptr %bc2, align 4
  %extract1 = extractvalue [2 x i32] %load2, 0
  %extract2 = extractvalue [2 x i32] %load2, 1

  ;CHECK-MATERIALIZER: load %"struct.std::array", ptr addrspace(4) @SpecConsBlob___test_kernel_1, align 1
  %gep3 = getelementptr i8, ptr %0, i32 18
  %bc3 = bitcast ptr %gep3 to ptr
  %load3 = load %"struct.std::array", ptr %bc3, align 1
  %D = extractvalue %"struct.std::array" %load3, 0, 2, 0
  %L = extractvalue %"struct.std::array" %load3, 0, 2, 1
  %H = extractvalue %"struct.std::array" %load3, 0, 2, 2

  %add1 = add nsw i32 %extract1, %load
  %add2 = add nsw i32 %add1, %extract2
  %conv1 = sitofp i32 %add2 to double

  %add3 = fadd double %D, %conv1
  %conv2 = sitofp i64 %L to double
  %add4 = fadd double %conv2, %add3
  %conv3 = fpext half %H to double
  %add5 = fadd double %conv3, %add4

  store double %add5, ptr addrspace(1) %out, align 4

  ret void
}

!SYCL_SpecConst_data = !{!1}
!1 = !{!"\07\00\00\00\03\00\00\00\01\00\00\00\06\00\00\00\04\00\00\00"}
