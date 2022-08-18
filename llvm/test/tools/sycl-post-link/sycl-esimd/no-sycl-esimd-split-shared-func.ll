; This test checks "integrated" invoke_simd support by the sycl-post-link tool:
; - library helper function used in _Z33__regcall3____builtin_invoke_simd* is
;   optimized
; - functions shared by SYCL and ESIMD callgraphs get cloned and renamed, thus
;   making sure no functions are shared by the callgraphs (currently required by
;   IGC)

; RUN: sycl-post-link -lower-esimd -symbols -split=auto -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.ll
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM

;---------------- Verify generated table file.
; CHECK-TABLE: [Code|Properties|Symbols]
; CHECK-TABLE: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-TABLE-EMPTY:

;---------------- Verify generated symbol file.
; CHECK-SYM: SPMD_CALLER
; CHECK-SYM: SYCL_kernel
; CHECK-SYM: ESIMD_kernel
; CHECK-SYM: SIMD_CALL_HELPER_{{[0-9]+}}
; CHECK-SYM-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent

declare dso_local spir_func <4 x float> @__intrin(i64)

;---- Function calleed from both SYCL and ESIMD callgraphs.
;---- Must be cloned for ESIMD.
define dso_local spir_func <4 x float> @SHARED_F(i64 %addr) noinline {
  %res = call spir_func <4 x float> @__intrin(i64 %addr)
  ret <4 x float> %res
}

declare dso_local x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (<4 x float> (<4 x float>)*, <4 x float>)* noundef, <4 x float> (<4 x float>)* noundef, float noundef)

;---- This has linkonce_odr, so it should be removed after inlining into the
;---- helper.
define linkonce_odr x86_regcallcc <4 x float> @SIMD_CALLEE(<4 x float> %val) #3 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK-NOT: {{.*}} @SIMD_CALLEE(
  %data = call spir_func <4 x float> @SHARED_F(i64 100)
  %add = fadd <4 x float> %val, %data
  ret <4 x float> %add
}

;---- Function containing the invoke_simd call.
define dso_local spir_func float @SPMD_CALLER(float %x) #0 !intel_reqd_sub_group_size !2 {
  %res = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (<4 x float> (<4 x float>)*, <4 x float>)* @SIMD_CALL_HELPER, <4 x float> (<4 x float>)* @SIMD_CALLEE, float %x)
  ret float %res
}

;---- Simd call helper library function mock.
define linkonce_odr dso_local x86_regcallcc <4 x float> @SIMD_CALL_HELPER(<4 x float> (<4 x float>)* noundef nonnull %f, <4 x float> %simd_args) #0 {
  %f.addr = alloca <4 x float> (<4 x float>)*, align 8
  %f.addr.ascast = addrspacecast <4 x float> (<4 x float>)** %f.addr to <4 x float> (<4 x float>)* addrspace(4)*
  store <4 x float> (<4 x float>)* %f, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast, align 8
  %1 = load <4 x float> (<4 x float>)*, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast, align 8
  %call = call x86_regcallcc <4 x float> %1(<4 x float> %simd_args)
  ret <4 x float> %call
}

;---- SYCL kernel, an entry point
define dso_local spir_kernel void @SYCL_kernel(float addrspace(1)* %ptr, float %x) #1 {
entry:
  %res1 = call spir_func float @SPMD_CALLER(float %x)
  %ptri = ptrtoint float addrspace(1)* %ptr to i64
  %vec = call spir_func <4 x float> @SHARED_F(i64 %ptri)
  %res2 = extractelement <4 x float> %vec, i32 0
  %res = fadd float %res1, %res2
  store float %res, float addrspace(1)* %ptr
  ret void
}

;---- ESIMD kernel, an entry point
define dso_local spir_kernel void @ESIMD_kernel(float addrspace(1)* %ptr) #2 !sycl_explicit_simd !0 {
entry:
  %ptr_as4 = addrspacecast float addrspace(1)* %ptr to float addrspace(4)*
  %res = call x86_regcallcc <4 x float> @SIMD_CALLEE(<4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  %ptr_x4 = bitcast float addrspace(4)* %ptr_as4 to <4 x float> addrspace(4)*
  store <4 x float> %res, <4 x float> addrspace(4)* %ptr_x4
  ret void
}

attributes #0 = { "sycl-module-id"="invoke_simd.cpp" }
attributes #1 = { "sycl-module-id"="a.cpp" }
attributes #2 = { "sycl-module-id"="b.cpp" }
attributes #3 = { "referenced-indirectly" }

!0 = !{}
!1 = !{i32 1}
!2 = !{i32 4}

;---------------- Verify IR. Outlined to avoid complications with reordering.
; Check the original version (for SYCL call graph) is retained
; CHECK: define dso_local spir_func <4 x float> @SHARED_F(

; Verify __builtin_invoke_simd lowering
; 1) the second argument (function pointer) is removed
; 2) The call target (helper) is changed to the optimized one
; CHECK: define dso_local spir_func float @SPMD_CALLER(float %{{.*}})
; CHECK:   %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX_{{.+}}(<4 x float> (<4 x float>)* @[[NEW_HELPER_NAME:SIMD_CALL_HELPER_[0-9]+]], float %{{.*}})
; CHECK:   ret float %{{.*}}

; Check the function is cloned for ESIMD call graph.
; CHECK: define dso_local spir_func <4 x float> @SHARED_F.esimd(i64 %{{.*}}) #[[SHARED_F_ATTRS:[0-9]+]] {
; CHECK:   %{{.*}} = call spir_func <4 x float> @__intrin(i64 %{{.*}})
; CHECK:   ret <4 x float> %{{.*}}

;---- Verify that the helper has been transformed and optimized:
;---- * %f removed
;---- * indirect call replaced with direct and inlined
;---- * linkonce_odr linkage replaced with weak_odr
;---- * sycl_explicit_simd and intel_reqd_sub_group_size attributes added, which
;----   is required for correct processing by LowerESIMD
; CHECK: define weak_odr dso_local x86_regcallcc <4 x float> @[[NEW_HELPER_NAME]](<4 x float> %{{.*}}) #[[NEW_HELPER_ATTRS:[0-9]+]] !sycl_explicit_simd !1 !intel_reqd_sub_group_size !2 {
; CHECK-NEXT:  %{{.*}} = call spir_func <4 x float> @SHARED_F.esimd(i64 100)
; CHECK-NEXT:  %{{.*}} = fadd <4 x float> %{{.*}}, %{{.*}}
; CHECK-NEXT:  ret <4 x float> {{.*}}

; Check that VCStackCall attribute is added to the invoke_simd helpers functions:
; CHECK: attributes #[[SHARED_F_ATTRS]] = { noinline "VCFunction" }
; CHECK: attributes #[[NEW_HELPER_ATTRS]] = { "VCFunction" "VCStackCall" "sycl-module-id"="invoke_simd.cpp" }

