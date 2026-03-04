; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-poison-stack-with-call=1 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id" = type { %"struct.user_def_types::no_cnstr" }
%"struct.user_def_types::no_cnstr" = type { float, i32, i8 }

@__usid_str = external addrspace(4) constant [57 x i8]
@_Z19spec_const_externalIN14user_def_types8no_cnstrELi1EE = external addrspace(1) constant %"class.sycl::_V1::specialization_id"

define spir_func i1 @_Z50check_kernel_handler_by_reference_external_handlerRN4sycl3_V114kernel_handlerEN14user_def_types8no_cnstrE() {
entry:
  %ref.tmp.i = alloca %"struct.user_def_types::no_cnstr", align 4
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
; CHECK: [[REG1:%[0-9]+]] = ptrtoint ptr addrspace(4) %ref.tmp.ascast.i to i64
; CHECK: call void @__msan_unpoison_shadow(i64 [[REG1]], i32 4, i64 12)
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14user_def_types8no_cnstrEET_PKcPKvS6_(ptr addrspace(4) dead_on_unwind writable sret(%"struct.user_def_types::no_cnstr") align 4 %ref.tmp.ascast.i, ptr addrspace(4) noundef @__usid_str, ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @_Z19spec_const_externalIN14user_def_types8no_cnstrELi1EE to ptr addrspace(4)), ptr addrspace(4) noundef null)
  ret i1 false
}

declare spir_func void @_Z40__sycl_getComposite2020SpecConstantValueIN14user_def_types8no_cnstrEET_PKcPKvS6_(ptr addrspace(4) sret(%"struct.user_def_types::no_cnstr"), ptr addrspace(4), ptr addrspace(4), ptr addrspace(4))
