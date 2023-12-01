// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: clang-offload-packager -o %t.out --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

//      OPENMP: @__start_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__stop_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__dummy.omp_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "omp_offloading_entries"
// OPENMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}"
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr @.omp_offloading.device_image, ptr getelementptr inbounds ([[[SIZE]] x i8], ptr @.omp_offloading.device_image, i64 1, i64 0), ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }]
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_reg, ptr null }]
// OPENMP-NEXT: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_unreg, ptr null }]

//      OPENMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

//      OPENMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// RUN: clang-offload-packager -o %t.out --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

//      CUDA: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".nv_fatbin"
// CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section ".nvFatBinSegment", align 8
// CUDA-NEXT: @__dummy.cuda_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "cuda_offloading_entries"
// CUDA-NEXT: @.cuda.binary_handle = internal global ptr null
// CUDA-NEXT: @__start_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @__stop_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.cuda.fatbin_reg, ptr null }]

//      CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = call ptr @__cudaRegisterFatBinary(ptr @.fatbin_wrapper)
// CUDA-NEXT:   store ptr %0, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @.cuda.globals_reg(ptr %0)
// CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(ptr %0)
// CUDA-NEXT:   %1 = call i32 @atexit(ptr @.cuda.fatbin_unreg)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

//      CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = load ptr, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @__cudaUnregisterFatBinary(ptr %0)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

//      CUDA: define internal void @.cuda.globals_reg(ptr %0) section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   br i1 icmp ne (ptr @__start_cuda_offloading_entries, ptr @__stop_cuda_offloading_entries), label %while.entry, label %while.end

//      CUDA: while.entry:
// CUDA-NEXT:  %entry1 = phi ptr [ @__start_cuda_offloading_entries, %entry ], [ %7, %if.end ]
// CUDA-NEXT:  %1 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 0
// CUDA-NEXT:  %addr = load ptr, ptr %1, align 8
// CUDA-NEXT:  %2 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 1
// CUDA-NEXT:  %name = load ptr, ptr %2, align 8
// CUDA-NEXT:  %3 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 2
// CUDA-NEXT:  %size = load i64, ptr %3, align 4
// CUDA-NEXT:  %4 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 3
// CUDA-NEXT:  %flag = load i32, ptr %4, align 4
// CUDA-NEXT:  %5 = icmp eq i64 %size, 0
// CUDA-NEXT:  br i1 %5, label %if.then, label %if.else

//      CUDA: if.then:
// CUDA-NEXT:   %6 = call i32 @__cudaRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// CUDA-NEXT:   br label %if.end

//      CUDA: if.else:
// CUDA-NEXT:   switch i32 %flag, label %if.end [
// CUDA-NEXT:     i32 0, label %sw.global
// CUDA-NEXT:     i32 1, label %sw.managed
// CUDA-NEXT:     i32 2, label %sw.surface
// CUDA-NEXT:     i32 3, label %sw.texture
// CUDA-NEXT:   ]

//      CUDA: sw.global:
// CUDA-NEXT:   call void @__cudaRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 0, i64 %size, i32 0, i32 0)
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.managed:
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.surface:
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.texture:
// CUDA-NEXT:   br label %if.end

//      CUDA: if.end:
// CUDA-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 1
// CUDA-NEXT:   %8 = icmp eq ptr %7, @__stop_cuda_offloading_entries
// CUDA-NEXT:   br i1 %8, label %while.end, label %while.entry

//      CUDA: while.end:
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// RUN: clang-offload-packager -o %t.out --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

//      HIP: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".hip_fatbin"
// HIP-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1212764230, i32 1, ptr @.fatbin_image, ptr null }, section ".hipFatBinSegment", align 8
// HIP-NEXT: @__dummy.hip_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "hip_offloading_entries"
// HIP-NEXT: @.hip.binary_handle = internal global ptr null
// HIP-NEXT: @__start_hip_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// HIP-NEXT: @__stop_hip_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// HIP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.hip.fatbin_reg, ptr null }]

//      HIP: define internal void @.hip.fatbin_reg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = call ptr @__hipRegisterFatBinary(ptr @.fatbin_wrapper)
// HIP-NEXT:   store ptr %0, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @.hip.globals_reg(ptr %0)
// HIP-NEXT:   %1 = call i32 @atexit(ptr @.hip.fatbin_unreg)
// HIP-NEXT:   ret void
// HIP-NEXT: }

//      HIP: define internal void @.hip.fatbin_unreg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = load ptr, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @__hipUnregisterFatBinary(ptr %0)
// HIP-NEXT:   ret void
// HIP-NEXT: }

//      HIP: define internal void @.hip.globals_reg(ptr %0) section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   br i1 icmp ne (ptr @__start_hip_offloading_entries, ptr @__stop_hip_offloading_entries), label %while.entry, label %while.end

//      HIP: while.entry:
// HIP-NEXT:   %entry1 = phi ptr [ @__start_hip_offloading_entries, %entry ], [ %7, %if.end ]
// HIP-NEXT:   %1 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 0
// HIP-NEXT:   %addr = load ptr, ptr %1, align 8
// HIP-NEXT:   %2 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 1
// HIP-NEXT:   %name = load ptr, ptr %2, align 8
// HIP-NEXT:   %3 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 2
// HIP-NEXT:   %size = load i64, ptr %3, align 4
// HIP-NEXT:   %4 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 3
// HIP-NEXT:   %flag = load i32, ptr %4, align 4
// HIP-NEXT:   %5 = icmp eq i64 %size, 0
// HIP-NEXT:   br i1 %5, label %if.then, label %if.else

//      HIP: if.then:
// HIP-NEXT:   %6 = call i32 @__hipRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// HIP-NEXT:   br label %if.end

//      HIP: if.else:
// HIP-NEXT:   switch i32 %flag, label %if.end [
// HIP-NEXT:     i32 0, label %sw.global
// HIP-NEXT:     i32 1, label %sw.managed
// HIP-NEXT:     i32 2, label %sw.surface
// HIP-NEXT:     i32 3, label %sw.texture
// HIP-NEXT:   ]

//      HIP: sw.global:
// HIP-NEXT:   call void @__hipRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 0, i64 %size, i32 0, i32 0)
// HIP-NEXT:   br label %if.end

//      HIP: sw.managed:
// HIP-NEXT:   br label %if.end

//      HIP: sw.surface:
// HIP-NEXT:   br label %if.end

//      HIP: sw.texture:
// HIP-NEXT:   br label %if.end

//      HIP: if.end:
// HIP-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 1
// HIP-NEXT:   %8 = icmp eq ptr %7, @__stop_hip_offloading_entries
// HIP-NEXT:   br i1 %8, label %while.end, label %while.entry

//      HIP: while.end:
// HIP-NEXT:   ret void
// HIP-NEXT: }

// TODO: get rid of test-sycl.o and prepare an input here.
// rUN: clang-offload-packager -o %t.out --image=file=%t.elf.o,kind=sycl,triple=spir64-unknown-unknown
// rUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// rUN:   -fembed-offload-object=%t.out

// TODO: Currently, linking part is failing. So we skip here an error temporarily.
// RUN: clang-linker-wrapper --print-wrapped-module --triple=spir64_unknown_unknown --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -sycl-post-link-options="-split=auto -symbols" -llvm-spirv-options="-spirv-max-version=1.4" \
// RUN:   -- %S/Inputs/test-sycl.o -o a.out 2>&1 >%t.output.txt || true
// RUN: FileCheck %s --check-prefix=SYCL <%t.output.txt

// SYCL-DAG: %_pi_device_binary_property_struct = type { ptr, ptr, i32, i64 }
// SYCL-DAG: %_pi_device_binary_property_set_struct = type { ptr, ptr, ptr }
// SYCL-DAG: %__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
// SYCL-DAG: %__tgt_device_image = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// SYCL-DAG: %__tgt_bin_desc.0 = type { i16, i16, ptr, ptr, ptr }

// SYCL-DAG: @.sycl_offloading.target.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-DAG: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-DAG: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-DAG: @prop = internal unnamed_addr constant [17 x i8] c"DeviceLibReqMask\00"
// SYCL-DAG: @__sycl_offload_prop_sets_arr = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop, ptr null, i32 1, i64 0 }]
// SYCL-DAG: @SYCL_PropSetName = internal unnamed_addr constant [24 x i8] c"SYCL/devicelib req mask\00"
// SYCL-DAG: @prop.1 = internal unnamed_addr constant [8 x i8] c"aspects\00"
// SYCL-DAG: @prop_val = internal unnamed_addr constant [8 x i8] zeroinitializer
// SYCL-DAG: @__sycl_offload_prop_sets_arr.2 = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop.1, ptr @prop_val, i32 2, i64 8 }]
// SYCL-DAG: @SYCL_PropSetName.3 = internal unnamed_addr constant [25 x i8] c"SYCL/device requirements\00"
// SYCL-DAG: @prop.4 = internal unnamed_addr constant [9 x i8] c"optLevel\00"
// SYCL-DAG: @__sycl_offload_prop_sets_arr.5 = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop.4, ptr null, i32 1, i64 2 }]
// SYCL-DAG: @SYCL_PropSetName.6 = internal unnamed_addr constant [21 x i8] c"SYCL/misc properties\00"
// SYCL-DAG: @__sycl_offload_prop_sets_arr.7 = internal constant [3 x %_pi_device_binary_property_set_struct] [%_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName, ptr @__sycl_offload_prop_sets_arr, ptr getelementptr inbounds ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr, i64 1, i64 0) }, %_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName.3, ptr @__sycl_offload_prop_sets_arr.2, ptr getelementptr inbounds ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr.2, i64 1, i64 0) }, %_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName.6, ptr @__sycl_offload_prop_sets_arr.5, ptr getelementptr inbounds ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr.5, i64 1, i64 0) }]
// SYCL-DAG: @.sycl_offloading.0.data = internal unnamed_addr constant
// SYCL-DAG: @__sycl_offload_entry_name = internal unnamed_addr constant
// SYCL-DAG: @__sycl_offload_entry_name.8 = internal unnamed_addr constant
// SYCL-DAG: @__sycl_offload_entries_arr = internal constant [2 x %__tgt_offload_entry] [%__tgt_offload_entry { ptr null, ptr @__sycl_offload_entry_name, i64 0, i32 0, i32 0 }, %__tgt_offload_entry { ptr null, ptr @__sycl_offload_entry_name.8, i64 0, i32 0, i32 0 }]
// SYCL-DAG: @.sycl_offloading.0.info = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr @.sycl_offloading.0.data to i64), i64 1160], section ".tgtimg", align 16
// SYCL-DAG: @llvm.used = appending global [1 x ptr] [ptr @.sycl_offloading.0.info], section "llvm.metadata"
// SYCL-DAG: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { i16 2, i8 4, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr null, ptr null, ptr @.sycl_offloading.0.data, ptr getelementptr inbounds ([1160 x i8], ptr @.sycl_offloading.0.data, i64 1, i64 0), ptr @__sycl_offload_entries_arr, ptr getelementptr inbounds ([2 x %__tgt_offload_entry], ptr @__sycl_offload_entries_arr, i64 1, i64 0), ptr @__sycl_offload_prop_sets_arr.7, ptr getelementptr inbounds ([3 x %_pi_device_binary_property_set_struct], ptr @__sycl_offload_prop_sets_arr.7, i64 1, i64 0) }]
// SYCL-DAG: @.sycl_offloading.descriptor = internal constant %__tgt_bin_desc.0 { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }
// SYCL-DAG: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_reg, ptr null }]
// SYCL-DAG: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_unreg, ptr null }]

//      SYCL: define internal void @sycl.descriptor_reg() section ".text.startup" {
// SYCL-NEXT: entry:
// SYCL-NEXT:   call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// SYCL-NEXT:   ret void
// SYCL-NEXT: }

//      SYCL: define internal void @sycl.descriptor_unreg() section ".text.startup" {
// SYCL-NEXT: entry:
// SYCL-NEXT:   call void @__sycl_unregister_lib(ptr @.sycl_offloading.descriptor)
// SYCL-NEXT:   ret void
// SYCL-NEXT: }
