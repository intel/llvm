; RUN: opt -passes=lower-invoke-simd -S < %s | FileCheck %s
; This test checks basic functionality of the LowerInvokeSimd pass:
; - __builtin_invoke_simd gets lowered as expected
; - actual call targets are successfully deduced in practical cases where they
;   are specified as function names in the user code
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZN4sycl3ext6oneapi12experimental6detail14unwrap_uniformIfE4implEf = comdat any

$SIMD_CALL_HELPER = comdat any

declare dso_local spir_func <16 x float> @__dummy_read() #4

define dso_local x86_regcallcc <16 x float> @SIMD_CALLEE(<16 x float> %x) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
  %y = call spir_func <16 x float> @__dummy_read()
  %z = fadd <16 x float> %x, %y
  ret <16 x float> %z
}

define dso_local x86_regcallcc <16 x float> @ANOTHER_SIMD_CALLEE(<16 x float> %x) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
  %y = call spir_func <16 x float> @__dummy_read()
  %z = fadd <16 x float> %x, %y
  ret <16 x float> %z
}

define dso_local spir_func noundef float @SPMD_CALLER(float noundef %x, ptr %raw_fptr) #0 {
; CHECK: define {{.*}} float @SPMD_CALLER(

;---- Typical data flow of the @SIMD_CALLEE function address in worst
;---- case (-O0), when invoke_simd uses function name:
;---- float res = invoke_simd(sg, SIMD_CALLEE, x);
  %f.addr.i = alloca ptr, align 8
  %f.addr.ascast.i = addrspacecast ptr %f.addr.i to ptr addrspace(4)
  store ptr @SIMD_CALLEE, ptr addrspace(4) %f.addr.ascast.i, align 8
  ;---- duplicated store of the same function pointer should be OK
  store ptr @SIMD_CALLEE, ptr addrspace(4) %f.addr.ascast.i, align 8
  %FUNC_PTR = load ptr, ptr addrspace(4) %f.addr.ascast.i, align 8

;---- The invoke_simd calls.
; Test case when function pointer (%FUNC_PTR) is passed the __builtin_invoke_simd,
; but the actual function can be deduced.
  %res1 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(ptr @SIMD_CALL_HELPER, ptr %FUNC_PTR, float %x)
; Verify that
; 1) the second argument (function pointer) is removed
; 2) The call target (helper) is changed to the optimized one
; CHECK: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX_{{.+}}(ptr @[[NAME1:SIMD_CALL_HELPER.+]], float %{{.*}})

; Test case when function name is passed directly to the __builtin_invoke_simd.
  %res2 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(ptr @SIMD_CALL_HELPER, ptr @ANOTHER_SIMD_CALLEE, float %x)
; CHECK: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX_{{.+}}(ptr @[[NAME2:SIMD_CALL_HELPER.+]], float %{{.*}})

; Test case when function pointer (%raw_fptr) is passed the __builtin_invoke_simd
; and actual function can't be deduced.
; Verify that there are no changes to the __builtin_invoke_simd call.
  %res3 = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(ptr @SIMD_CALL_HELPER, ptr %raw_fptr,  float %x)
; CHECK: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(ptr @SIMD_CALL_HELPER, ptr %{{.*}}, float %{{.*}})

  %res4 = fadd float %res1, %res2
  %res = fadd float %res3, %res4
  ret float %res
}
; CHECK: }

;---- Simd call helper library function mock.
define linkonce_odr dso_local x86_regcallcc <16 x float> @SIMD_CALL_HELPER(ptr noundef nonnull %f, <16 x float> %simd_args) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
  %f.addr = alloca ptr, align 8
  %f.addr.ascast = addrspacecast ptr %f.addr to ptr addrspace(4)
  store ptr %f, ptr addrspace(4) %f.addr.ascast, align 8
  %1 = load ptr, ptr addrspace(4) %f.addr.ascast, align 8
  %call = call x86_regcallcc <16 x float> %1(<16 x float> %simd_args)
  ret <16 x float> %call
}

;---- Check that original SIMD_CALL_HELPER retained, because there are
;---- invoke_simd calls where simd target can't be inferred.
; CHECK: define weak_odr {{.*}} <16 x float> @SIMD_CALL_HELPER(ptr {{.*}}%{{.*}}, <16 x float> %{{.*}}) #[[HELPER_ATTRS:[0-9]+]] !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1
; CHECK:   %{{.*}} = call x86_regcallcc <16 x float> %{{.*}}(<16 x float> %{{.*}})
; CHECK: }

;---- Optimized version for the SIMD_CALLEE call
; CHECK: define weak_odr {{.*}} <16 x float> @[[NAME1]](<16 x float> %{{.*}}) #[[HELPER_ATTRS1:[0-9]+]]
; Verify that indirect call is converted to direct
; CHECK: %{{.*}} = call x86_regcallcc <16 x float> @SIMD_CALLEE(<16 x float> %{{.*}})
; CHECK: }

;---- Optimized version for the ANOTHER_SIMD_CALLEE call
; CHECK: define weak_odr {{.*}} <16 x float> @[[NAME2]](<16 x float> %{{.*}}) #[[HELPER_ATTRS1]]
; Verify that indirect call is converted to direct
; CHECK: %{{.*}} = call x86_regcallcc <16 x float> @ANOTHER_SIMD_CALLEE(<16 x float> %{{.*}})
; CHECK: }

declare dso_local x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdXX(ptr noundef, ptr noundef, float noundef)

; Check that VCStackCall attribute is added to the invoke_simd target functions:
attributes #0 = { "sycl-module-id"="invoke_simd.cpp" }
; CHECK: attributes #[[HELPER_ATTRS]] = { "VCStackCall" "sycl-module-id"="invoke_simd.cpp" }
; If we transformed the helper, then it should receive "referenced-indirectly"
; attribute so it is not dropped after Internalize + DCE in post-split module
; cleanup
; CHECK: attributes #[[HELPER_ATTRS1]] = { "VCStackCall" "referenced-indirectly" "sycl-module-id"="invoke_simd.cpp" }

!0 = !{}
!1 = !{i32 16}
