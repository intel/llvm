; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_variable_length_array
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv  --spec-const=0:i64:28 -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; The IR was generated from the following source:
; #include <CL/sycl.hpp>
; #include <cstdint>
;
; class SpecializedKernel;
; class MyUInt64Const;
;
; // Fetch a value at runtime.
; SYCL_EXTERNAL
; uint64_t get_value();
;
; int main() {
;   cl::sycl::queue queue;
;   cl::sycl::program program(queue.get_context());
;
;   // Create specialization constants.
;   cl::sycl::experimental::spec_constant<uint64_t, MyUInt64Const> spec_const =
;       program.set_spec_constant<MyUInt64Const>(get_value());
;
;   program.build_with_kernel_type<SpecializedKernel>();
;
;   {
;     queue.submit([&](cl::sycl::handler &cgh) {
;       cgh.single_task<SpecializedKernel>(
;           program.get_kernel<SpecializedKernel>(),
;           [=]() {
;              int foo_arr[spec_const.get()];
;              foo_arr[0] = 42;
;           });
;     });
;   }
; }
; Command line(clang and sycl-post-link from https://github.com/intel/llvm):
; clang -fsycl -fsycl-device-only vla_spec_const.cpp -c
; sycl-post-link vla_spec_const.bc -spec-const=rt
; llvm-dis vla_spec_const_0.bc -o vla_spec_const.ll

; CHECK-SPIRV: Capability VariableLengthArrayINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_variable_length_array"

; CHECK-SPIRV: Decorate [[#SPEC_CONST:]] SpecId 0
; CHECK-SPIRV: SpecConstant [[#]] [[#SPEC_CONST]]

; CHECK-SPIRV-LABEL: FunctionEnd
; CHECK-SPIRV: FunctionCall [[#]] [[#SPEC_CONST_VAL:]] [[#SPEC_CONST_GET:]]
; CHECK-SPIRV: SaveMemoryINTEL
; CHECK-SPIRV: VariableLengthArrayINTEL [[#]] [[#]] [[#SPEC_CONST_VAL]]
; CHECK-SPIRV: RestoreMemoryINTEL

; CHECK-SPIRV: Function [[#]] [[#SPEC_CONST_GET]]
; CHECK-SPIRV: ReturnValue [[#SPEC_CONST]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type { %"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" }
%"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" = type { i8 }

$_ZTS17SpecializedKernel = comdat any

$_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv = comdat any

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTS17SpecializedKernel() #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 1
  %1 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  %2 = addrspacecast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this) #2 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, align 8
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8, !tbaa !5
  %this1 = load %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8
  %0 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this1, i32 0, i32 0
  %call = call spir_func i64 @_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv(%"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" addrspace(4)* %0)
; CHECK-LLVM:  %[[SPEC_CONST:[[:alnum:]]+]] = call spir_func i64 @_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv(
  %1 = call i8* @llvm.stacksave()
; CHECK-LLVM: call i8* @llvm.stacksave()
  store i8* %1, i8** %saved_stack, align 8
  %vla = alloca i32, i64 %call, align 4
; CHECK-LLVM: alloca i32, i64 %[[SPEC_CONST]], align 4
  store i64 %call, i64* %__vla_expr0, align 8
  %ptridx = getelementptr inbounds i32, i32* %vla, i64 0
  store i32 42, i32* %ptridx, align 4, !tbaa !9
  %2 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %2)
; CHECK-LLVM: call void @llvm.stackrestore(i8*
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl12experimental13spec_constantIm13MyUInt64ConstE3getEv(%"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" addrspace(4)* %this) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" addrspace(4)*, align 8
  %TName = alloca i8 addrspace(4)*, align 8
  store %"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" addrspace(4)* %this, %"class._ZTSN2cl4sycl12experimental13spec_constantIm13MyUInt64ConstEE.cl::sycl::experimental::spec_constant" addrspace(4)** %this.addr, align 8, !tbaa !5
  %0 = bitcast i8 addrspace(4)** %TName to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #4
  %1 = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 0), !SYCL_SPEC_CONST_SYM_ID !11
  %2 = bitcast i8 addrspace(4)** %TName to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %2) #4
; CHECK-LLVM: ret i64 28
  ret i64 %1
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #4

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #4

declare i64 @_Z20__spirv_SpecConstantix(i32, i64)

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/work/intel/vla_spec_const.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 (https://github.com/intel/llvm.git fdc14595bd27ca70472ca719e383935e25de6710)"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!"_ZTS13MyUInt64Const", i32 0}
