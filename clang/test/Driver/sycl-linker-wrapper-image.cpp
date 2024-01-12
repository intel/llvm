// REQUIRES: system-linux
// This test check wrapping of SYCL binaries in clang-linker-wrapper.

// RUN: %clang -cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.device.bc
// RUN: clang-offload-packager -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu --triple=spir64 \
// RUN:                      -sycl-device-library-location=%S/Inputs -sycl-post-link-options="-split=auto -symbols" \
// RUN:                      %t.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
    func();
}

extern "C" {
// symbols so that linker find them and doesn't fail.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
    kernel<class fake_kernel>([](){});
}

//#endif

// CHECK-DAG: %_pi_device_binary_property_struct = type { ptr, ptr, i32, i64 }
// CHECK-DAG: %_pi_device_binary_property_set_struct = type { ptr, ptr, ptr }
// CHECK-DAG: %struct.__tgt_offload_entry = type { ptr, ptr, i64, i32, i32 }
// CHECK-DAG: %__sycl.tgt_device_image = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-DAG: %__sycl.tgt_bin_desc = type { i16, i16, ptr, ptr, ptr }

// CHECK-DAG: @.sycl_offloading.target.0 = internal unnamed_addr constant [23 x i8] c"spir64-unknown-unknown\00"
// CHECK-DAG: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-DAG: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-DAG: @prop = internal unnamed_addr constant [17 x i8] c"DeviceLibReqMask\00"
// CHECK-DAG: @__sycl_offload_prop_sets_arr = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop, ptr null, i32 1, i64 0 }]
// CHECK-DAG: @SYCL_PropSetName = internal unnamed_addr constant [24 x i8] c"SYCL/devicelib req mask\00"
// CHECK-DAG: @prop.1 = internal unnamed_addr constant [8 x i8] c"aspects\00"
// CHECK-DAG: @prop_val = internal unnamed_addr constant [8 x i8] zeroinitializer
// CHECK-DAG: @__sycl_offload_prop_sets_arr.2 = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop.1, ptr @prop_val, i32 2, i64 8 }]
// CHECK-DAG: @SYCL_PropSetName.3 = internal unnamed_addr constant [25 x i8] c"SYCL/device requirements\00"
// CHECK-DAG: @__sycl_offload_prop_sets_arr.4 = internal constant [2 x %_pi_device_binary_property_set_struct] [%_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName, ptr @__sycl_offload_prop_sets_arr, ptr getelementptr inbounds ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr, i64 1, i64 0) }, %_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName.3, ptr @__sycl_offload_prop_sets_arr.2, ptr getelementptr inbounds ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr.2, i64 1, i64 0) }]
// CHECK-DAG: @.sycl_offloading.0.data = internal unnamed_addr constant [740 x i8] 
// CHECK-DAG: @__sycl_offload_entry_name = internal unnamed_addr constant [25 x i8] c"_ZTSZ4mainE11fake_kernel\00"
// CHECK-DAG: @__sycl_offload_entries_arr = internal constant [1 x %struct.__tgt_offload_entry] [%struct.__tgt_offload_entry { ptr null, ptr @__sycl_offload_entry_name, i64 0, i32 0, i32 0 }]
// CHECK-DAG: @.sycl_offloading.0.info = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr @.sycl_offloading.0.data to i64), i64 740], section ".tgtimg", align 16
// CHECK-DAG: @llvm.used = appending global [1 x ptr] [ptr @.sycl_offloading.0.info], section "llvm.metadata"
// CHECK-DAG: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 2, i8 4, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr null, ptr null, ptr @.sycl_offloading.0.data, ptr getelementptr inbounds ([740 x i8], ptr @.sycl_offloading.0.data, i64 1, i64 0), ptr @__sycl_offload_entries_arr, ptr getelementptr inbounds ([1 x %struct.__tgt_offload_entry], ptr @__sycl_offload_entries_arr, i64 1, i64 0), ptr @__sycl_offload_prop_sets_arr.4, ptr getelementptr inbounds ([2 x %_pi_device_binary_property_set_struct], ptr @__sycl_offload_prop_sets_arr.4, i64 1, i64 0) }]
// CHECK-DAG: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }
// CHECK-DAG: @llvm.global_ctors = {{.*}} { i32 1, ptr @sycl.descriptor_reg, ptr null }]
// CHECK-DAG: @llvm.global_dtors = {{.*}} { i32 1, ptr @sycl.descriptor_unreg, ptr null }]

//      CHECK: define internal void @sycl.descriptor_reg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

//      CHECK: define internal void @sycl.descriptor_unreg() section ".text.startup" {
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_unregister_lib(ptr @.sycl_offloading.descriptor)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
