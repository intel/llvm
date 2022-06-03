; RUN: opt -passes=lower-invoke-simd -S < %s | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent
declare dso_local spir_func <16 x float> @__dummy_read(i64) #4

; Function Attrs: convergent
declare dso_local spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)*, float addrspace(4)*, float, i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local x86_regcallcc <16 x float> @_SIMD_CALLEE(float addrspace(4)* %A, <16 x float> %non_uni_val, i32 %uni_val) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !0 {
; Verify that correct attributes are attached to the function:
; CHECK: {{.*}} @_SIMD_CALLEE(float addrspace(4)* %A, <16 x float> %non_uni_val, i32 %uni_val) #0
entry:
  %AA = ptrtoint float addrspace(4)* %A to i64
  %ii = zext i32 %uni_val to i64
  %addr = add nuw nsw i64 %ii, %AA
  %data = call spir_func <16 x float> @__dummy_read(i64 %addr)
  %add = fadd <16 x float> %non_uni_val, %data
  ret <16 x float> %add
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define dso_local x86_regcallcc <16 x float> @_ANOTHER_SIMD_CALLEE(float addrspace(4)* %A, <16 x float> %non_uni_val, i32 %uni_val) #1 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !0 {
; Verify that correct attributes are attached to the function:
; CHECK: {{.*}} @_ANOTHER_SIMD_CALLEE(float addrspace(4)* %A, <16 x float> %non_uni_val, i32 %uni_val) #1
entry:
  %AA = ptrtoint float addrspace(4)* %A to i64
  %ii = zext i32 %uni_val to i64
  %addr = add nuw nsw i64 %ii, %AA
  %data = call spir_func <16 x float> @__dummy_read(i64 %addr)
  %add = fadd <16 x float> %non_uni_val, %data
  ret <16 x float> %add
}

define internal spir_func float @foo(float addrspace(1)* %ptr, <16 x float> (float addrspace(4)*, <16 x float>, i32)* %raw_fptr) align 2 {
entry:
;------------- Typical data flow of the @_SIMD_CALLEE function address in worst
;------------- case (-O0), when invoke_simd uses function name:
;------------- float res = invoke_simd(sg, SIMD_CALLEE, uniform{ A }, x, uniform{ y });
  %f.addr.i = alloca <16 x float> (float addrspace(4)*, <16 x float>, i32)*, align 8
  %f.addr.ascast.i = addrspacecast <16 x float> (float addrspace(4)*, <16 x float>, i32)** %f.addr.i to <16 x float> (float addrspace(4)*, <16 x float>, i32)* addrspace(4)*
  store <16 x float> (float addrspace(4)*, <16 x float>, i32)* @_SIMD_CALLEE, <16 x float> (float addrspace(4)*, <16 x float>, i32)* addrspace(4)* %f.addr.ascast.i, align 8
  %FUNC_PTR = load <16 x float> (float addrspace(4)*, <16 x float>, i32)*, <16 x float> (float addrspace(4)*, <16 x float>, i32)* addrspace(4)* %f.addr.ascast.i, align 8

;------------- Data flow for the parameters of SIMD_CALLEE
  %param_A = addrspacecast float addrspace(1)* %ptr to float addrspace(4)*
  %param_non_uni_val = load float, float addrspace(4)* %param_A, align 4

;------------- The invoke_simd calls.
  %res1 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)* %FUNC_PTR, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)
; Verify that %FUNC_PTR is replaced with @_SIMD_CALLEE:
; CHECK: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)* @_SIMD_CALLEE, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)

  %res2 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)* @_ANOTHER_SIMD_CALLEE, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)
; Verify that function address link-time constant is accepted by the pass and left as is:
; CHECK: = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)* @_ANOTHER_SIMD_CALLEE, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)

; TODO: enable in the test and LowerInvokeSimd when BE is ready, crash for now:
  ;%res3 %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<16 x float> (float addrspace(4)*, <16 x float>, i32)* %raw_fptr, float addrspace(4)* %param_A, float %param_non_uni_val, i32 10)
  %res = fadd float %res1, %res2
  ret float %res
}

; Check that VCStackCall attribute is added to the invoke_simd target functions:
attributes #0 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="512" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="invoke_simd.cpp" }
; CHECK: attributes #0 = { convergent mustprogress norecurse "VCStackCall" {{.*}} "sycl-module-id"="invoke_simd.cpp" }
attributes #1 = { convergent mustprogress norecurse "sycl-module-id"="invoke_simd.cpp" }
; CHECK: attributes #1 = { convergent mustprogress norecurse "VCStackCall" "sycl-module-id"="invoke_simd.cpp" }

!0 = !{}
!1 = !{i32 16}
