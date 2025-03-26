// REQUIRES: system-linux
// This test check wrapping of SYCL binaries in clang-linker-wrapper.
//
// Generate .o file as linker wrapper input.
//
// touch %t.device.bc
// RUN: clang-offload-packager -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t.devicelib.o
//
// Run clang-linker-wrapper test and check the output of SYCL Offload Wrapping.
//
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:                      -sycl-device-libraries=%t.devicelib.o \
// RUN:                       %t.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s

// CHECK: %_pi_device_binary_property_struct = type { ptr, ptr, i32, i64 }
// CHECK-NEXT: %_pi_device_binary_property_set_struct = type { ptr, ptr, ptr }
// CHECK-NEXT: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK-NEXT: %__sycl.tgt_device_image = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-NEXT: %__sycl.tgt_bin_desc = type { i16, i16, ptr, ptr, ptr }

// CHECK: @.sycl_offloading.target.0 = internal unnamed_addr constant [7 x i8] c"spir64\00"
// CHECK-NEXT: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-NEXT: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-NEXT: @prop = internal unnamed_addr constant [4 x i8] c"key\00"
// CHECK-NEXT: @__sycl_offload_prop_sets_arr = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop, ptr null, i32 1, i64 0 }]
// CHECK-NEXT: @SYCL_PropSetName = internal unnamed_addr constant [25 x i8] c"SYCL/device requirements\00"
// CHECK-NEXT: @__sycl_offload_prop_sets_arr.1 = internal constant [1 x %_pi_device_binary_property_set_struct] [%_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName, ptr @__sycl_offload_prop_sets_arr, ptr getelementptr ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr, i64 0, i64 1) }]
// CHECK-NEXT: @.sycl_offloading.0.data = internal unnamed_addr constant [0 x i8] zeroinitializer, section "spir64"
// CHECK-NEXT: @__sycl_offload_entry_name = internal unnamed_addr constant [7 x i8] c"entry1\00"
// CHECK-NEXT: @__sycl_offload_entry_name.2 = internal unnamed_addr constant [7 x i8] c"entry2\00"
// CHECK-NEXT: @__sycl_offload_entries_arr = internal constant [2 x %struct.__tgt_offload_entry] [%struct.__tgt_offload_entry { i64 0, i16 1, i16 4, i32 0, ptr null, ptr @__sycl_offload_entry_name, i64 0, i64 0, ptr null }, %struct.__tgt_offload_entry { i64 0, i16 1, i16 4, i32 0, ptr null, ptr @__sycl_offload_entry_name.2, i64 0, i64 0, ptr null }]
// CHECK-NEXT: @.sycl_offloading.0.info = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr @.sycl_offloading.0.data to i64), i64 0], section ".tgtimg", align 16
// CHECK-NEXT: @llvm.used = appending global [1 x ptr] [ptr @.sycl_offloading.0.info], section "llvm.metadata"
// CHECK-NEXT: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 2, i8 4, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr null, ptr null, ptr @.sycl_offloading.0.data, ptr @.sycl_offloading.0.data, ptr @__sycl_offload_entries_arr, ptr getelementptr ([2 x %struct.__tgt_offload_entry], ptr @__sycl_offload_entries_arr, i64 0, i64 2), ptr @__sycl_offload_prop_sets_arr.1, ptr getelementptr ([1 x %_pi_device_binary_property_set_struct], ptr @__sycl_offload_prop_sets_arr.1, i64 0, i64 1) }]
// CHECK-NEXT: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }
// CHECK-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_reg, ptr null }]
// CHECK-NEXT: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_unreg, ptr null }]

// CHECK: define internal void @sycl.descriptor_reg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK: define internal void @sycl.descriptor_unreg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_unregister_lib(ptr @.sycl_offloading.descriptor)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
