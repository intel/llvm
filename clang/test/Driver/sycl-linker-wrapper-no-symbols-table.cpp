// REQUIRES: system-linux
//
// Regression test: -sycl-thin-lto makes sycl-post-link emit a 2-column
// "[Code|Properties]" table (no -symbols), which must be accepted.
//
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.device.bc
// RUN: llvm-offload-binary -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
//
// Generate .bc file as SYCL device library file.
//
// RUN: touch %t.devicelib.bc
//
// Run clang-linker-wrapper with -sycl-thin-lto (no explicit -symbols in
// -sycl-post-link-options) so sycl-post-link actually runs and produces a
// Symbols-less table; clang-linker-wrapper must parse it successfully.
//
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:                      --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:                      -sycl-thin-lto -sycl-post-link-options="-split=auto" -sycl-post-link-options="-properties" %t.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s

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

// CHECK-NOT: invalid SYCL Table file.
// CHECK-DAG: @.sycl_offloading.target.0 = internal unnamed_addr constant [7 x i8] c"spir64\00"
// CHECK-DAG: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image]
// CHECK-DAG: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }
