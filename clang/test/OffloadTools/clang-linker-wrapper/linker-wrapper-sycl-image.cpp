// REQUIRES: system-linux, x86-registered-target, spirv-to-ir-wrapper, sycl-post-link

// Check the SYCL image, properties, entries, and registration emitted by
// clang-linker-wrapper. Also check that thin LTO accepts a sycl-post-link table
// without a Symbols column.

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.device.bc
// RUN: llvm-offload-binary -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang_cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
// RUN: touch %t.devicelib.bc

// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties --linker-path=/usr/bin/ld \
// RUN:   %t.o -o %t.out 2>&1 | FileCheck %s

// With --emit-fatbin-only, the output is an offload container holding the
// finalized SPIR-V with its runtime metadata. The host compilation wraps it
// into the same descriptor-based registration.
// RUN: clang-linker-wrapper --emit-fatbin-only --no-sycl-rdc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties --linker-path=/usr/bin/ld \
// RUN:   %t.o -o %t.syclfb
// RUN: llvm-objdump --offloading %t.syclfb | FileCheck %s --check-prefix=NORDC
// NORDC: OFFLOADING IMAGE [0]:
// NORDC: triple spir64-unknown-unknown
// NORDC: producer sycl
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.syclfb -emit-llvm %s -o - | FileCheck %s

// With thin LTO, sycl-post-link emits a two-column [Code|Properties] table.
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-thin-lto -sycl-post-link-options=-split=auto \
// RUN:   -sycl-post-link-options=-properties --linker-path=/usr/bin/ld \
// RUN:   %t.o -o %t.thin-lto.out 2>&1 | FileCheck %s --check-prefix=THIN-LTO

// Device pre-linking must produce an object without runtime references when
// there are no device inputs or when the device module is empty.
// RUN: %clang_cc1 -DEMPTY -triple=x86_64-unknown-linux-gnu %s -emit-obj -o %t.host-only.o
// RUN: clang-linker-wrapper --sycl-device-link --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.host-only.o -o %t.empty.o
// RUN: %clang %t.host-only.o %t.empty.o -o %t.host-only.out
// RUN: clang-linker-wrapper --sycl-device-link --print-wrapped-module \
// RUN:   --host-triple=x86_64-pc-windows-msvc --linker-path=/usr/bin/ld \
// RUN:   %t.host-only.o -o %t.empty.obj 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: %clang_cc1 -DEMPTY -fsycl-is-device -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.empty.bc
// RUN: llvm-offload-binary -o %t.empty.fat --image=file=%t.empty.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang_cc1 -DEMPTY -triple=x86_64-unknown-linux-gnu %s -emit-obj -o %t.empty-input.o -fembed-offload-object=%t.empty.fat
// RUN: clang-linker-wrapper --sycl-device-link --print-wrapped-module \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties %t.empty-input.o -o %t.empty.o 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: %clang %t.host-only.o %t.empty.o -o %t.empty.out
// RUN: clang-linker-wrapper --sycl-device-link --no-use-sycl-post-link-tool \
// RUN:   --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   %t.empty-input.o -o %t.empty.o 2>&1 | FileCheck %s --check-prefix=EMPTY
// RUN: %clang %t.host-only.o %t.empty.o -o %t.empty-library.out

// Empty no-RDC images still need a container for the host compilation.
// RUN: clang-linker-wrapper --emit-fatbin-only --no-sycl-rdc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties %t.empty-input.o -o %t.empty.syclfb
// RUN: %clang_cc1 -DEMPTY -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.empty.syclfb -emit-llvm %s -o - | FileCheck %s --check-prefix=EMPTY
// RUN: %clang_cc1 -DEMPTY -triple x86_64-pc-windows-msvc -fsycl-is-host \
// RUN:   -foffload-include-binary %t.empty.syclfb -emit-llvm %s -o - | FileCheck %s --check-prefix=EMPTY
// RUN: clang-linker-wrapper --emit-fatbin-only --no-sycl-rdc --no-use-sycl-post-link-tool \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   %t.empty-input.o -o %t.empty-library.syclfb
// RUN: %clang_cc1 -DEMPTY -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.empty-library.syclfb -emit-llvm %s -o - | FileCheck %s --check-prefix=EMPTY

// An empty target must not add a placeholder when another target has images.
// RUN: llvm-offload-binary -o %t.mixed.fat --image=file=%t.empty.bc,kind=sycl,triple=spir64-unknown-unknown --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown,arch=generic
// RUN: clang-linker-wrapper --emit-fatbin-only --no-sycl-rdc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties %t.mixed.fat -o %t.mixed.syclfb
// RUN: llvm-objdump --offloading %t.mixed.syclfb | FileCheck %s --check-prefix=MIXED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.mixed.syclfb -emit-llvm %s -o - | FileCheck %s
// MIXED: OFFLOADING IMAGE [0]:
// MIXED: producer sycl
// MIXED-NOT: OFFLOADING IMAGE
// EMPTY: target triple
// EMPTY-NOT: __sycl_register_lib
// EMPTY-NOT: __sycl_unregister_lib

#ifndef EMPTY
template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
  func();
}

// Provide the registration symbols needed by the host linker.
extern "C" {
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
  kernel<class fake_kernel>([]() {});
}
#else
int main() {}
#endif

// THIN-LTO-NOT: invalid SYCL Table file.
// THIN-LTO: @.sycl_offloading.target.0 = internal unnamed_addr constant [7 x i8] c"spir64\00"
// THIN-LTO: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image]
// THIN-LTO: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }

// CHECK-DAG: %_pi_device_binary_property_struct = type { ptr, ptr, i32, i64 }
// CHECK-DAG: %_pi_device_binary_property_set_struct = type { ptr, ptr, ptr }
// CHECK-DAG: %struct.__tgt_offload_entry = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK-DAG: %__sycl.tgt_device_image = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-DAG: %__sycl.tgt_bin_desc = type { i16, i16, ptr, ptr, ptr }

// CHECK-DAG: @.sycl_offloading.target.0 = internal unnamed_addr constant [7 x i8] c"spir64\00"
// CHECK-DAG: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-DAG: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-DAG: @prop = internal unnamed_addr constant [8 x i8] c"aspects\00"
// CHECK-DAG: @prop_val = internal unnamed_addr constant [8 x i8] zeroinitializer
// CHECK-DAG: @__sycl_offload_prop_sets_arr = internal constant [1 x %_pi_device_binary_property_struct] [%_pi_device_binary_property_struct { ptr @prop, ptr @prop_val, i32 2, i64 8 }]
// CHECK-DAG: @SYCL_PropSetName = internal unnamed_addr constant [25 x i8] c"SYCL/device requirements\00"
// CHECK-DAG: @SYCL_PropSetName.1 = internal unnamed_addr constant [22 x i8] c"SYCL/kernel param opt\00"
// CHECK-DAG: @__sycl_offload_prop_sets_arr.2 = internal constant [2 x %_pi_device_binary_property_set_struct] [%_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName, ptr @__sycl_offload_prop_sets_arr, ptr getelementptr ([1 x %_pi_device_binary_property_struct], ptr @__sycl_offload_prop_sets_arr, i64 0, i64 1) }, %_pi_device_binary_property_set_struct { ptr @SYCL_PropSetName.1, ptr null, ptr null }]
// CHECK-DAG: @.sycl_offloading.0.data = internal unnamed_addr constant [[[SIZE1:.*]] x i8]
// CHECK-DAG: @__sycl_offload_entry_name = internal unnamed_addr constant [25 x i8] c"_ZTSZ4mainE11fake_kernel\00"
// CHECK-DAG: @__sycl_offload_entries_arr = internal constant [1 x %struct.__tgt_offload_entry] [%struct.__tgt_offload_entry { i64 0, i16 1, i16 8, i32 0, ptr null, ptr @__sycl_offload_entry_name, i64 0, i64 0, ptr null }]
// CHECK-DAG: @.sycl_offloading.0.info = internal local_unnamed_addr constant [2 x i64] [i64 ptrtoint (ptr @.sycl_offloading.0.data to i64), i64 772], section ".tgtimg", align 16
// CHECK-DAG: @llvm.used = appending global [1 x ptr] [ptr @.sycl_offloading.0.info], section "llvm.metadata"

// CHECK-DAG: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 3, i8 4, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr @.sycl_offloading.0.data, ptr getelementptr ([[[SIZE2:.*]] x i8], ptr @.sycl_offloading.0.data, i64 0, i64 [[SIZE2]]), ptr @__sycl_offload_entries_arr, ptr getelementptr ([1 x %struct.__tgt_offload_entry], ptr @__sycl_offload_entries_arr, i64 0, i64 1), ptr @__sycl_offload_prop_sets_arr.2, ptr getelementptr ([2 x %_pi_device_binary_property_set_struct], ptr @__sycl_offload_prop_sets_arr.2, i64 0, i64 2) }]
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
