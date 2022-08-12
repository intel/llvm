; This test checks "integrated" invoke_simd support by the sycl-post-link tool:
; - data flow analysis is able to determine actual target of
;   _Z33__regcall3____builtin_invoke_simd*
; - functions shared by SYCL and ESIMD callgraphs get cloned and renamed, thus
;   making sure no functions are shared by the callgraphs (currently required by
;   IGC)

; RUN: sycl-post-link -lower-esimd -symbols -split=auto -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM

;------------- Verify generated table file.
; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-EMPTY:

;------------- Verify generated symbol file.
; CHECK-SYM: SYCL_kernel
; CHECK-SYM: SIMD_CALLEE
; CHECK-SYM: ESIMD_kernel
; CHECK-SYM-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent

declare dso_local spir_func <4 x float> @__intrin(i64)

define dso_local spir_func <4 x float> @SHARED_FUNCTION(i64 %addr) noinline {
;------------- This function is participates in both SYCL and ESIMD callgraphs -
;------------- check that it is duplicated:
; CHECK-IR-DAG: define dso_local spir_func <4 x float> @SHARED_FUNCTION.esimd
  %res = call spir_func <4 x float> @__intrin(i64 %addr)
  ret <4 x float> %res
}

declare dso_local x86_regcallcc noundef float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (<4 x float> (<4 x float>)*, <4 x float>)* noundef, <4 x float> (<4 x float>)* noundef, float noundef)

;------------- This is also an entry point, because of the "sycl-module-id" attribute #0.
define linkonce_odr x86_regcallcc <4 x float> @SIMD_CALLEE(<4 x float> %val) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !0 {
; Verify that correct attributes are attached to the function:
; CHECK-IR-DAG: {{.*}} @SIMD_CALLEE(<4 x float> %{{.*}}) #[[ATTR1:[0-9]+]]
  %data = call spir_func <4 x float> @SHARED_FUNCTION(i64 100)
  %add = fadd <4 x float> %val, %data
  ret <4 x float> %add
}

define dso_local spir_func float @SPMD_CALLER(float %x) #0 align 2 {
;---- Typical data flow of the @SIMD_CALLEE function address in worst
;---- case (-O0), when invoke_simd uses function name:
;---- float res = invoke_simd(sg, SIMD_CALLEE, x);
  %f.addr.i = alloca <4 x float> (<4 x float>)*, align 8
  %f.addr.ascast.i = addrspacecast <4 x float> (<4 x float>)** %f.addr.i to <4 x float> (<4 x float>)* addrspace(4)*
  store <4 x float> (<4 x float>)* @SIMD_CALLEE, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast.i, align 8
  %FUNC_PTR = load <4 x float> (<4 x float>)*, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast.i, align 8

;---- The invoke_simd call.
; Test case when function pointer (%FUNC_PTR) is passed the __builtin_invoke_simd,
; but the actual function can be deduced.
  %res = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX(<4 x float> (<4 x float> (<4 x float>)*, <4 x float>)* @SIMD_CALL_HELPER, <4 x float> (<4 x float>)* %FUNC_PTR, float %x)
; Verify that
; 1) the second argument (function pointer) is removed
; 2) The call target (helper) is changed to the optimized one
; CHECK: %{{.*}} = call spir_func float @_Z33__regcall3____builtin_invoke_simdXX_{{.+}}(<4 x float> (<4 x float>)* @[[NAME1:SIMD_CALL_HELPER.+]], float %{{.*}})

  ret float %res
}
; CHECK: }

;---- Simd call helper library function mock.
define linkonce_odr dso_local x86_regcallcc <4 x float> @SIMD_CALL_HELPER(<4 x float> (<4 x float>)* noundef nonnull %f, <4 x float> %simd_args) #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
  %f.addr = alloca <4 x float> (<4 x float>)*, align 8
  %f.addr.ascast = addrspacecast <4 x float> (<4 x float>)** %f.addr to <4 x float> (<4 x float>)* addrspace(4)*
  store <4 x float> (<4 x float>)* %f, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast, align 8
  %1 = load <4 x float> (<4 x float>)*, <4 x float> (<4 x float>)* addrspace(4)* %f.addr.ascast, align 8
  %call = call x86_regcallcc <4 x float> %1(<4 x float> %simd_args)
  ret <4 x float> %call
}

;---- Output optimized version for the SIMD_CALLEE call
; CHECK: define {{.*}} <4 x float> @[[NAME1]](<4 x float> %{{.*}}) #1
; Verify that indirect call is converted to direct
; CHECK: %{{.*}} = call x86_regcallcc <4 x float> @SIMD_CALLEE(<4 x float> %{{.*}})
; CHECK: }

;------------- SYCL kernel, an entry point
define dso_local spir_kernel void @SYCL_kernel(float addrspace(1)* %ptr, float %x) #1 {
entry:
  %res1 = call spir_func float @SPMD_CALLER(float %x)
  %ptri = ptrtoint float addrspace(1)* %ptr to i64
  %vec = call spir_func <4 x float> @SHARED_FUNCTION(i64 %ptri)
  %res2 = extractelement <4 x float> %vec, i32 0
  %res = fadd float %res1, %res2
  store float %res, float addrspace(1)* %ptr
  ret void
}

;------------- ESIMD kernel, an entry point
define dso_local spir_kernel void @ESIMD_kernel(float addrspace(1)* %ptr) #2 !sycl_explicit_simd !0 {
entry:
  %ptr_as4 = addrspacecast float addrspace(1)* %ptr to float addrspace(4)*
  %res = call x86_regcallcc <4 x float> @SIMD_CALLEE(<4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  %ptr_x4 = bitcast float addrspace(4)* %ptr_as4 to <4 x float> addrspace(4)*
  store <4 x float> %res, <4 x float> addrspace(4)* %ptr_x4
  ret void
}

; Check that VCStackCall attribute is added to the invoke_simd helpers functions:
attributes #0 = { "sycl-module-id"="invoke_simd.cpp" }
; CHECK-IR-DAG: attributes #[[ATTR1]] = { "VCStackCall" {{.*}}"sycl-module-id"="invoke_simd.cpp" }

attributes #1 = { "sycl-module-id"="a.cpp" }
attributes #2 = { "sycl-module-id"="b.cpp" }


attributes #0 = { noinline }
attributes #1 = { "sycl-module-id"="invoke_simd.cpp" }
attributes #2 = { "sycl-module-id"="a.cpp" }
attributes #3 = { noinline "VCFunction" }
attributes #4 = { alwaysinline "VCFunction" "sycl-module-id"="invoke_simd.cpp" }
attributes #5 = { "CMGenxMain" "VCFunction" "VCNamedBarrierCount"="0" "VCSLMSize"="0" "oclrt"="1" "sycl-module-id"="b.cpp" }
attributes #6 = { nounwind readnone "VCFunction" }
attributes #7 = { "VCFunction" "VCStackCall" "sycl-module-id"="invoke_simd.cpp" }


!0 = !{}
!1 = !{i32 16}
