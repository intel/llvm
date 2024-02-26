; LLVM IR for this test is produced from the following SYCL code snippet:
;
; #include <sycl/sycl.hpp>
;
; struct user_defined_type {
;   float a;
;   int b;
;   char c;
;
;   constexpr user_defined_type(float a, int b, char c) : a(a), b(b), c(c) {}
; };
;
; constexpr sycl::specialization_id<user_defined_type> spec_id(3.14, 42, 8);
;
; int main() {
;   sycl::queue q;
;   user_defined_type data(0, 0, 0);
;   sycl::buffer buf(&data, sycl::range<1>{1});
;   q.submit([&](sycl::handler &cgh) {
;     auto acc = buf.get_access();
;     cgh.single_task([=](sycl::kernel_handler kh) {
;       acc[0] = kh.get_specialization_constant<spec_id>();
;     });
;   });
;
;   return 0;
; }
;
; Compiled with: clang++ -fsycl -fsycl-device-only -O2 -emit-llvm -S -fno-sycl-instrument-device-code
;
; 'user_defined_type' is taken from SYCL-CTS for spec constants.
;
; The idea of the test is to ensure that SpecConstants pass is able to handle
; a situation, where spec constant default value contains less elements than
; spec constant type, due to padding inserted by a compiler.
;
; RUN: sycl-post-link --spec-const=native -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll
; RUN: sycl-post-link -debug-only=SpecConst --spec-const=native -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOG
;
; CHECK: %[[#A:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#ID:]], float 0x40091EB860000000)
; CHECK: %[[#B:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID+1]], i32 42)
; CHECK: %[[#C:]] = call i8 @_Z20__spirv_SpecConstantia(i32 2, i8 8)
; CHECK: call %struct.user_defined_type @_Z29__spirv_SpecConstantCompositefiaA3_a_Rstruct.user_defined_type(float %[[#A]], i32 %[[#B]], i8 %[[#C]], [3 x i8] undef)
;
; CHECK: !sycl.specialization-constants = !{![[#SC:]]}
; CHECK: ![[#SC]] = !{!"uidac684fbd602505be____ZL7spec_id",
; CHECK-SAME: i32 [[#ID]], i32 0, i32 4
; CHECK-SAME: i32 [[#ID+1]], i32 4, i32 4
; CHECK-SAME: i32 [[#ID+2]], i32 8, i32 1
; CHECK-SAME: i32 -1, i32 9, i32 3
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 9, 3}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG:{0, 4, 3.140000e+00}
; CHECK-LOG:{4, 4, 42}
; CHECK-LOG:{8, 1, 8}

source_filename = "t.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.user_defined_type = type <{ float, i32, i8, [3 x i8] }>
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_14kernel_handlerEE_ = comdat any

@__usid_str = private unnamed_addr constant [34 x i8] c"uidac684fbd602505be____ZL7spec_id\00", align 1
@_ZL7spec_id = internal addrspace(1) constant { { float, i32, i8 } } { { float, i32, i8 } { float 0x40091EB860000000, i32 42, i8 8 } }, align 4

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_14kernel_handlerEE_(%struct.user_defined_type addrspace(1)* noundef align 4 %_arg_acc, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %_arg_acc3) local_unnamed_addr #0 comdat !srcloc !48 !kernel_arg_buffer_location !49 !kernel_arg_runtime_aligned !50 !kernel_arg_exclusive_ptr !50 !sycl_fixed_targets !51 !sycl_kernel_omit_args !52 {
entry:
  %ref.tmp.i = alloca %struct.user_defined_type, align 4
  %0 = bitcast %"class.sycl::_V1::id"* %_arg_acc3 to i64*
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds %struct.user_defined_type, %struct.user_defined_type addrspace(1)* %_arg_acc, i64 %1
  %ref.tmp.ascast.i = addrspacecast %struct.user_defined_type* %ref.tmp.i to %struct.user_defined_type addrspace(4)*
  %2 = bitcast %struct.user_defined_type* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %2) #4
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI17user_defined_typeET_PKcPKvS5_(%struct.user_defined_type addrspace(4)* sret(%struct.user_defined_type) align 4 %ref.tmp.ascast.i, i8 addrspace(4)* noundef addrspacecast (i8* getelementptr inbounds ([34 x i8], [34 x i8]* @__usid_str, i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast ({ { float, i32, i8 } } addrspace(1)* @_ZL7spec_id to i8 addrspace(1)*) to i8 addrspace(4)*), i8 addrspace(4)* noundef null) #5
  %3 = bitcast %struct.user_defined_type addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %4 = bitcast %struct.user_defined_type* %ref.tmp.i to i8*
  call void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* align 4 %3, i8* align 4 %4, i64 9, i1 false), !tbaa.struct !53
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %2) #4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI17user_defined_typeET_PKcPKvS5_(%struct.user_defined_type addrspace(4)* sret(%struct.user_defined_type) align 4, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef, i8 addrspace(4)* noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #3

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="t.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46}
!llvm.ident = !{!47}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"cpu", i32 1}
!5 = !{!"gpu", i32 2}
!6 = !{!"accelerator", i32 3}
!7 = !{!"custom", i32 4}
!8 = !{!"fp16", i32 5}
!9 = !{!"fp64", i32 6}
!10 = !{!"image", i32 9}
!11 = !{!"online_compiler", i32 10}
!12 = !{!"online_linker", i32 11}
!13 = !{!"queue_profiling", i32 12}
!14 = !{!"usm_device_allocations", i32 13}
!15 = !{!"usm_host_allocations", i32 14}
!16 = !{!"usm_shared_allocations", i32 15}
!17 = !{!"usm_system_allocations", i32 17}
!18 = !{!"ext_intel_pci_address", i32 18}
!19 = !{!"ext_intel_gpu_eu_count", i32 19}
!20 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!21 = !{!"ext_intel_gpu_slices", i32 21}
!22 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!23 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!24 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!25 = !{!"ext_intel_mem_channel", i32 25}
!26 = !{!"usm_atomic_host_allocations", i32 26}
!27 = !{!"usm_atomic_shared_allocations", i32 27}
!28 = !{!"atomic64", i32 28}
!29 = !{!"ext_intel_device_info_uuid", i32 29}
!30 = !{!"ext_oneapi_srgb", i32 30}
!31 = !{!"ext_oneapi_native_assert", i32 31}
!32 = !{!"host_debuggable", i32 32}
!33 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!34 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!35 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!36 = !{!"ext_intel_free_memory", i32 36}
!37 = !{!"ext_intel_device_id", i32 37}
!38 = !{!"ext_intel_memory_clock_rate", i32 38}
!39 = !{!"ext_intel_memory_bus_width", i32 39}
!40 = !{!"emulated", i32 40}
!41 = !{!"ext_intel_legacy_image", i32 41}
!42 = !{!"int64_base_atomics", i32 7}
!43 = !{!"int64_extended_atomics", i32 8}
!44 = !{!"usm_system_allocator", i32 17}
!45 = !{!"usm_restricted_shared_allocations", i32 16}
!46 = !{!"host", i32 0}
!47 = !{!"clang version 17.0.0 (https://github.com/intel/llvm.git)"}
!48 = !{i32 443}
!49 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!50 = !{i1 true, i1 false, i1 false, i1 false, i1 false}
!51 = !{}
!52 = !{i1 false, i1 true, i1 true, i1 false, i1 true}
!53 = !{i64 0, i64 4, !54, i64 4, i64 4, !58, i64 8, i64 1, !60}
!54 = !{!55, !55, i64 0}
!55 = !{!"float", !56, i64 0}
!56 = !{!"omnipotent char", !57, i64 0}
!57 = !{!"Simple C++ TBAA"}
!58 = !{!59, !59, i64 0}
!59 = !{!"int", !56, i64 0}
!60 = !{!56, !56, i64 0}
