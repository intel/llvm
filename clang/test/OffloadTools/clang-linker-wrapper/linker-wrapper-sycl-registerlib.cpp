// REQUIRES: system-linux, x86-registered-target, spirv-to-ir-wrapper, sycl-post-link

// Check that the SYCL wrapper object for a target defines the
// __sycl_registerlib_<id> symbols referenced by the host objects whose device
// code is for that target, including archive members found through -l and old
// offload model objects.

// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.bc
// RUN: llvm-offload-binary -o %t.fat --image=file=%t.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: llvm-offload-binary -o %t.gen.fat --image=file=%t.bc,kind=sycl,triple=spir64_gen-unknown-unknown,arch=pvc
// RUN: %clang_cc1 -DMAIN %s -triple=x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=main -emit-obj -fembed-offload-object=%t.fat -o %t.main.o
// RUN: %clang_cc1 -DFN=lib_fn %s -triple=x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=lib -emit-obj -fembed-offload-object=%t.fat -o %t.lib.o
// RUN: %clang_cc1 -DFN=gen_fn %s -triple=x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=gen -emit-obj -fembed-offload-object=%t.gen.fat -o %t.gen.o
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: llvm-ar rc %t.dir/libsyclreg.a %t.lib.o %t.gen.o
// RUN: %clang_cc1 -DFN=old_fn %s -triple=x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=old -emit-obj -fembed-offload-object=%t.bc -o %t.old.host.o
// RUN: %clang_cc1 -DFN=oldgen_fn %s -triple=x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=oldgen -emit-obj -fembed-offload-object=%t.bc -o %t.oldgen.host.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown -input=%t.old.host.o -input=%t.bc -output=%t.old.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-pvc -input=%t.oldgen.host.o -input=%t.bc -output=%t.oldgen.o
// RUN: llvm-ar rc %t.dir/libsyclold.a %t.old.o %t.oldgen.o
// RUN: touch %t.devicelib.bc
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   -sycl-post-link-options=-split=auto -sycl-post-link-options=-symbols \
// RUN:   -sycl-post-link-options=-properties --linker-path=/usr/bin/ld \
// RUN:   %t.main.o -L%t.dir -lsyclreg -lsyclold -o %t.out 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=__sycl_registerlib_gen \
// RUN:   --implicit-check-not=__sycl_registerlib_oldgen

// No spir64_gen device code is linked, so nothing defines that reference.
// CHECK: define weak void @__sycl_registerlib_main()
// CHECK: define weak void @__sycl_registerlib_lib()
// CHECK: define weak void @__sycl_registerlib_old()

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
  func();
}

#ifdef MAIN
extern "C" {
// Define stub versions of the SYCL runtime image registration functions that are called by
// the wrapped host code so we don't need to link against the real SYCL runtime.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

void lib_fn();
int main() {
  kernel<class main_kernel>([]() {});
  lib_fn();
}
#else
void FN() { kernel<class lib_kernel>([]() {}); }
#endif
