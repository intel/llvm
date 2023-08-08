; NOTE: address-space argument is crucial since we test SPIRV code.
; RUN: opt -S -passes=infer-address-spaces --address-space=4 %s | FileCheck %s

; Check that InferAddressSpacesPass doesn't removes metadata attached to GEPs.
; CHECK: getelementptr {{.*}} !llvm.index.group

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.my_kernel = type { i32 addrspace(4)* }

$_ZTS9my_kernel = comdat any

@.str = private unnamed_addr addrspace(1) constant [34 x i8] c"{memory:DEFAULT}{sizeinfo:4,5,10}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [9 x i8] c"code.cpp\00", section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS9my_kernel(i32 addrspace(1)* noundef align 4 %_arg_a) #0 comdat !srcloc !59 !kernel_arg_buffer_location !60 !sycl_fixed_targets !61 {
entry:
  %this.addr.i = alloca %struct.my_kernel addrspace(4)*, align 8
  %line_buffer.i = alloca [5 x [10 x i32]], align 4
  %i.i = alloca i32, align 4
  %cleanup.dest.slot.i = alloca i32, align 4
  %j.i = alloca i32, align 4
  %_arg_a.addr = alloca i32 addrspace(1)*, align 8
  %my_kernel = alloca %struct.my_kernel, align 8
  %_arg_a.addr.ascast = addrspacecast i32 addrspace(1)** %_arg_a.addr to i32 addrspace(1)* addrspace(4)*
  %my_kernel.ascast = addrspacecast %struct.my_kernel* %my_kernel to %struct.my_kernel addrspace(4)*
  store i32 addrspace(1)* %_arg_a, i32 addrspace(1)* addrspace(4)* %_arg_a.addr.ascast, align 8, !tbaa !62
  %0 = bitcast %struct.my_kernel* %my_kernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #3
  %a = getelementptr inbounds %struct.my_kernel, %struct.my_kernel addrspace(4)* %my_kernel.ascast, i32 0, i32 0
  %1 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %_arg_a.addr.ascast, align 8, !tbaa !62
  %2 = addrspacecast i32 addrspace(1)* %1 to i32 addrspace(4)*
  store i32 addrspace(4)* %2, i32 addrspace(4)* addrspace(4)* %a, align 8, !tbaa !66
  %this.addr.ascast.i = addrspacecast %struct.my_kernel addrspace(4)** %this.addr.i to %struct.my_kernel addrspace(4)* addrspace(4)*
  %line_buffer.ascast.i = addrspacecast [5 x [10 x i32]]* %line_buffer.i to [5 x [10 x i32]] addrspace(4)*
  %i.ascast.i = addrspacecast i32* %i.i to i32 addrspace(4)*
  %cleanup.dest.slot.ascast.i = addrspacecast i32* %cleanup.dest.slot.i to i32 addrspace(4)*
  %j.ascast.i = addrspacecast i32* %j.i to i32 addrspace(4)*
  store %struct.my_kernel addrspace(4)* %my_kernel.ascast, %struct.my_kernel addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !62
  %this1.i = load %struct.my_kernel addrspace(4)*, %struct.my_kernel addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %3 = bitcast [5 x [10 x i32]]* %line_buffer.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 200, i8* %3) #3
  %line_buffer.ascast2.i = bitcast [5 x [10 x i32]] addrspace(4)* %line_buffer.ascast.i to i8 addrspace(4)*
  %line_buffer.ascast3.i = addrspacecast i8 addrspace(4)* %line_buffer.ascast2.i to i8*
  call void @llvm.var.annotation.p0i8.p1i8(i8* %line_buffer.ascast3.i, i8 addrspace(1)* getelementptr inbounds ([34 x i8], [34 x i8] addrspace(1)* @.str, i32 0, i32 0), i8 addrspace(1)* getelementptr inbounds ([9 x i8], [9 x i8] addrspace(1)* @.str.1, i32 0, i32 0), i32 11, i8 addrspace(1)* null)
  %4 = bitcast i32* %i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #3
  store i32 0, i32 addrspace(4)* %i.ascast.i, align 4, !tbaa !68
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.cond.cleanup6.i, %entry
  %5 = load i32, i32 addrspace(4)* %i.ascast.i, align 4, !tbaa !68
  %cmp.i = icmp slt i32 %5, 5
  br i1 %cmp.i, label %for.body.i, label %_ZNK9my_kernelclEv.exit

for.body.i:                                       ; preds = %for.cond.i
  %6 = bitcast i32* %j.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #3
  store i32 0, i32 addrspace(4)* %j.ascast.i, align 4, !tbaa !68
  br label %for.cond4.i

for.cond4.i:                                      ; preds = %for.body7.i, %for.body.i
  %7 = load i32, i32 addrspace(4)* %j.ascast.i, align 4, !tbaa !68
  %cmp5.i = icmp slt i32 %7, 5
  br i1 %cmp5.i, label %for.body7.i, label %for.cond.cleanup6.i

for.cond.cleanup6.i:                              ; preds = %for.cond4.i
  store i32 5, i32 addrspace(4)* %cleanup.dest.slot.ascast.i, align 4
  %8 = bitcast i32* %j.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %8) #3
  %9 = load i32, i32 addrspace(4)* %i.ascast.i, align 4, !tbaa !68
  %inc11.i = add nsw i32 %9, 1
  store i32 %inc11.i, i32 addrspace(4)* %i.ascast.i, align 4, !tbaa !68
  br label %for.cond.i, !llvm.loop !70

for.body7.i:                                      ; preds = %for.cond4.i
  %a.i = getelementptr inbounds %struct.my_kernel, %struct.my_kernel addrspace(4)* %this1.i, i32 0, i32 0
  %10 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %a.i, align 8, !tbaa !66
  %11 = load i32, i32 addrspace(4)* %10, align 4, !tbaa !68
  %12 = load i32, i32 addrspace(4)* %i.ascast.i, align 4, !tbaa !68
  %idxprom.i = sext i32 %12 to i64
  %arrayidx.i = getelementptr inbounds [5 x [10 x i32]], [5 x [10 x i32]] addrspace(4)* %line_buffer.ascast.i, i64 0, i64 %idxprom.i, !llvm.index.group !74
  %13 = load i32, i32 addrspace(4)* %j.ascast.i, align 4, !tbaa !68
  %idxprom8.i = sext i32 %13 to i64
  %arrayidx9.i = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %arrayidx.i, i64 0, i64 %idxprom8.i
  store i32 %11, i32 addrspace(4)* %arrayidx9.i, align 4, !tbaa !68
  %14 = load i32, i32 addrspace(4)* %j.ascast.i, align 4, !tbaa !68
  %inc.i = add nsw i32 %14, 1
  store i32 %inc.i, i32 addrspace(4)* %j.ascast.i, align 4, !tbaa !68
  br label %for.cond4.i, !llvm.loop !76

_ZNK9my_kernelclEv.exit:                          ; preds = %for.cond.i
  store i32 2, i32 addrspace(4)* %cleanup.dest.slot.ascast.i, align 4
  %15 = bitcast i32* %i.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %15) #3
  %16 = bitcast [5 x [10 x i32]]* %line_buffer.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 200, i8* %16) #3
  %17 = bitcast %struct.my_kernel* %my_kernel to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %17) #3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.var.annotation.p0i8.p1i8(i8*, i8 addrspace(1)*, i8 addrspace(1)*, i32, i8 addrspace(1)*) #2

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="code.cpp" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57}
!llvm.ident = !{!58}

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
!42 = !{!"ext_oneapi_bindless_images", i32 42}
!43 = !{!"ext_oneapi_bindless_images_shared_usm", i32 43}
!44 = !{!"ext_oneapi_bindless_images_1d_usm", i32 44}
!45 = !{!"ext_oneapi_bindless_images_2d_usm", i32 45}
!46 = !{!"ext_oneapi_interop_memory_import", i32 46}
!47 = !{!"ext_oneapi_interop_memory_export", i32 47}
!48 = !{!"ext_oneapi_interop_semaphore_import", i32 48}
!49 = !{!"ext_oneapi_interop_semaphore_export", i32 49}
!50 = !{!"ext_oneapi_mipmap", i32 50}
!51 = !{!"ext_oneapi_mipmap_anisotropy", i32 51}
!52 = !{!"ext_oneapi_mipmap_level_reference", i32 52}
!53 = !{!"int64_base_atomics", i32 7}
!54 = !{!"int64_extended_atomics", i32 8}
!55 = !{!"usm_system_allocator", i32 17}
!56 = !{!"usm_restricted_shared_allocations", i32 16}
!57 = !{!"host", i32 0}
!58 = !{!"clang version 17.0.0 (https://github.com/intel/llvm.git 8e0cc4b7a845df9389a1313a3e680babc4d87782)"}
!59 = !{i32 84}
!60 = !{i32 -1}
!61 = !{}
!62 = !{!63, !63, i64 0}
!63 = !{!"any pointer", !64, i64 0}
!64 = !{!"omnipotent char", !65, i64 0}
!65 = !{!"Simple C++ TBAA"}
!66 = !{!67, !63, i64 0}
!67 = !{!"_ZTS9my_kernel", !63, i64 0}
!68 = !{!69, !69, i64 0}
!69 = !{!"int", !64, i64 0}
!70 = distinct !{!70, !71, !72}
!71 = !{!"llvm.loop.mustprogress"}
!72 = !{!"llvm.loop.parallel_access_indices", !73}
!73 = distinct !{}
!74 = !{!73, !75}
!75 = distinct !{}
!76 = distinct !{!76, !71, !77}
!77 = !{!"llvm.loop.parallel_access_indices", !75}

