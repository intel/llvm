; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck --check-prefix CHECK-LLVM %s

; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#I32]] [[#CONST_I32_3:]] 3
; CHECK-SPIRV-DAG: Constant [[#I32]] [[#CONST_I32_8:]] 8
; CHECK-SPIRV-DAG: TypeFloat [[#HALF:]] 16
; CHECK-SPIRV-DAG: TypeStruct [[#S_HALF:]] [[#HALF]]
; CHECK-SPIRV-DAG: TypePointer [[#PTR_S_HALF:]] {{[0-9]+}} [[#S_HALF]]

target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::detail::half_impl::half" = type { half }

define spir_func void @test_group_non_uniform_shuffle_down() {
entry:
  %agg.tmp.i.i = alloca %"class.sycl::_V1::detail::half_impl::half", align 2
  %ref.tmp.i = alloca %"class.sycl::_V1::detail::half_impl::half", align 2
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  call spir_func void @_Z34__spirv_GroupNonUniformShuffleDownIN4sycl3_V16detail9half_impl4halfEET_N5__spv5Scope4FlagES5_j(ptr addrspace(4) dead_on_unwind writable sret(%"class.sycl::_V1::detail::half_impl::half") align 2 %ref.tmp.ascast.i, i32 noundef 3, ptr noundef nonnull byval(%"class.sycl::_V1::detail::half_impl::half") align 2 %agg.tmp.i.i, i32 noundef 8)
  ret void
}

; CHECK-SPIRV: Variable {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: Variable [[#PTR_S_HALF]] [[#VAR_0:]]
; CHECK-SPIRV: Load [[#S_HALF]] [[#COMP_0:]] [[#VAR_0]]
; CHECK-SPIRV: CompositeExtract [[#HALF]] [[#ELEM_0:]] [[#COMP_0]] 0
; CHECK-SPIRV: GroupNonUniformShuffleDown [[#HALF]] [[#ELEM_1:]] [[#CONST_I32_3]] [[#ELEM_0]] [[#CONST_I32_8]]
; CHECK-SPIRV: CompositeInsert [[#S_HALF]] [[#COMP_1:]] [[#ELEM_1]] [[#COMP_0]] 0
; CHECK-SPIRV: Store [[#VAR_0]] [[#COMP_1]]

; CHECK-LLVM: [[ALLOCA_0:%[a-z0-9.]+]] = alloca %"class.sycl::_V1::detail::half_impl::half", align 2
; CHECK-LLVM: [[ALLOCA_1:%[a-z0-9.]+]] = alloca %"class.sycl::_V1::detail::half_impl::half", align 2
; CHECK-LLVM: [[LOAD_0:%[a-z0-9.]+]] = load %"class.sycl::_V1::detail::half_impl::half", ptr [[ALLOCA_1]], align 2
; CHECK-LLVM: [[EXTRACT_0:%[a-z0-9.]+]] = extractvalue %"class.sycl::_V1::detail::half_impl::half" [[LOAD_0]], 0
; CHECK-LLVM: [[CALL_0:%[a-z0-9.]+]] = call spir_func half @_Z22sub_group_shuffle_downDhj(half [[EXTRACT_0]], i32 8) #0
; CHECK-LLVM: [[INSERT_0:%[a-z0-9.]+]] = insertvalue %"class.sycl::_V1::detail::half_impl::half" [[LOAD_0]], half [[CALL_0]], 0
; CHECK-LLVM: store %"class.sycl::_V1::detail::half_impl::half" [[INSERT_0]], ptr [[ALLOCA_1]], align 2

declare dso_local spir_func void @_Z34__spirv_GroupNonUniformShuffleDownIN4sycl3_V16detail9half_impl4halfEET_N5__spv5Scope4FlagES5_j(ptr addrspace(4) dead_on_unwind writable sret(%"class.sycl::_V1::detail::half_impl::half") align 2, i32 noundef, ptr noundef byval(%"class.sycl::_V1::detail::half_impl::half") align 2, i32 noundef) local_unnamed_addr