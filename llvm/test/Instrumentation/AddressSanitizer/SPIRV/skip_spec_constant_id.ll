; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

; check that specialization constant ids (sycl::specialization_id) are not
; instrumented, since sycl-post-link reads their default values from the
; initializers.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { i64 }
%"class.sycl::_V1::specialization_id.0" = type { %struct.custom_type }
%struct.custom_type = type { i32, float }

@__usid_str = private unnamed_addr addrspace(1) constant [5 x i8] c"uid0\00", align 1
@__usid_str.1 = private unnamed_addr addrspace(1) constant [5 x i8] c"uid1\00", align 1
@__usid_str.2 = private unnamed_addr addrspace(1) constant [5 x i8] c"uid2\00", align 1
@__usid_str.3 = private unnamed_addr addrspace(1) constant [5 x i8] c"uid3\00", align 1
@__usid_str.4 = private unnamed_addr addrspace(1) constant [5 x i8] c"uid4\00", align 1
@_ZL9alloca_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 8 }, align 8
@_ZL9scalar_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 4 }, align 8
@_ZL12composite_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.0" { %struct.custom_type { i32 1, float 2.000000e+00 } }, align 4
; Passed through an addrspacecast instruction.
@_ZL7cast_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 2 }, align 8
; Not constant, e.g. the value type has a mutable member.
@_ZL10mutable_id = internal addrspace(1) global %"class.sycl::_V1::specialization_id.0" { %struct.custom_type { i32 3, float 4.000000e+00 } }, align 4
@global_var = internal addrspace(1) constant i64 16, align 8
; Passed to a getter, but not as the spec id.
@rt_buffer = internal addrspace(1) constant i64 0, align 8

; CHECK: @_ZL9alloca_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 8 }, no_sanitize_address
; CHECK: @_ZL9scalar_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 4 }, no_sanitize_address
; CHECK: @_ZL12composite_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.0" { %struct.custom_type { i32 1, float 2.000000e+00 } }, no_sanitize_address
; CHECK: @_ZL7cast_id = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 2 }, no_sanitize_address
; CHECK: @_ZL10mutable_id = internal addrspace(1) global %"class.sycl::_V1::specialization_id.0" { %struct.custom_type { i32 3, float 4.000000e+00 } }, no_sanitize_address
; Other globals are still instrumented.
; CHECK: @global_var = internal addrspace(1) constant { i64, [24 x i8] }
; CHECK: @rt_buffer = internal addrspace(1) constant { i64, [24 x i8] }
; CHECK: @__AsanDeviceGlobalMetadata{{.*}} = {{.*}} [2 x { i64, i64, i64 }]

define spir_kernel void @test(ptr addrspace(1) %out) #0 {
entry:
  %alloca = call ptr @llvm.sycl.alloca.p0.p4.p4.p4.i64(ptr addrspace(4) addrspacecast (ptr addrspace(1) @__usid_str to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL9alloca_id to ptr addrspace(4)), ptr addrspace(4) null, i64 0, i64 4)
  %scalar = call spir_func i64 @_Z37__sycl_getScalar2020SpecConstantValueImET_PKcPKvS4_(ptr addrspace(4) addrspacecast (ptr addrspace(1) @__usid_str.1 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL9scalar_id to ptr addrspace(4)), ptr addrspace(4) null)
  %composite = alloca %struct.custom_type, align 4
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI11custom_typeET_PKcPKvS5_(ptr sret(%struct.custom_type) align 4 %composite, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__usid_str.2 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL12composite_id to ptr addrspace(4)), ptr addrspace(4) null)
  %cast = addrspacecast ptr addrspace(1) @_ZL7cast_id to ptr addrspace(4)
  %scalar2 = call spir_func i64 @_Z37__sycl_getScalar2020SpecConstantValueImET_PKcPKvS4_(ptr addrspace(4) addrspacecast (ptr addrspace(1) @__usid_str.3 to ptr addrspace(4)), ptr addrspace(4) %cast, ptr addrspace(4) addrspacecast (ptr addrspace(1) @rt_buffer to ptr addrspace(4)))
  %mutable = alloca %struct.custom_type, align 4
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI11custom_typeET_PKcPKvS5_(ptr sret(%struct.custom_type) align 4 %mutable, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__usid_str.4 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL10mutable_id to ptr addrspace(4)), ptr addrspace(4) null)
  %val = load i64, ptr addrspace(1) @global_var, align 8
  %sum = add i64 %scalar, %val
  %sum2 = add i64 %sum, %scalar2
  store i64 %sum2, ptr addrspace(1) %out, align 8
  ret void
}

declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.i64(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), i64, i64 immarg)
declare spir_func i64 @_Z37__sycl_getScalar2020SpecConstantValueImET_PKcPKvS4_(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))
declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI11custom_typeET_PKcPKvS5_(ptr sret(%struct.custom_type) align 4, ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))

attributes #0 = { sanitize_address }
