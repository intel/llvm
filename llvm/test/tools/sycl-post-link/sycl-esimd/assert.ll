; RUN: sycl-post-link -properties -split-esimd  -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.prop

; Verify we mark a image with an ESIMD kernel with the isEsimdImage property

; CHECK: isEsimdImage=1|1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"struct.sycl::_V1::detail::AssertHappened" = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

@.str = private unnamed_addr addrspace(1) constant [10 x i8] c"Id != 400\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [8 x i8] c"foo.cpp\00", align 1
@__PRETTY_FUNCTION__ = private unnamed_addr addrspace(1) constant [56 x i8] c"auto main()::(anonymous class)::operator()(id<1>) const\00", align 1
@SPIR_AssertHappenedMem = linkonce_odr dso_local addrspace(1) global %"struct.sycl::_V1::detail::AssertHappened" zeroinitializer, align 8

declare void @llvm.assume(i1 noundef) #2

define weak_odr dso_local spir_kernel void @esimd_kernel() local_unnamed_addr #0 !sycl_explicit_simd !0 {
entry:
  tail call spir_func void @__assert_fail(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4)), i32 noundef 13, ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__PRETTY_FUNCTION__ to ptr addrspace(4))) #12
  ret void
}

define weak dso_local spir_func void @__assert_fail(ptr addrspace(4) noundef %expr, ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %func) #1 {
entry:
  tail call spir_func void @__devicelib_assert_fail(ptr addrspace(4) noundef %expr, ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %func) #1
  ret void
}

define weak dso_local spir_func void @__devicelib_assert_fail(ptr addrspace(4) noundef %expr, ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %func) #2 {
entry:                                 
  ret void
}

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="foo.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="bar.cpp" "sycl-optlevel"="2" }
attributes #2 = { convergent nounwind }

!0 = !{}
