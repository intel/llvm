// REQUIRES: system-linux, x86-registered-target, sycl-post-link, llvm-spirv

// Test the "no-rdc" linking flow for device code compiled with --offload-arch
// for IGCA targets.

// RUN: split-file %s %t
// RUN: llvm-as %t/resolved.ll -o %t/resolved.bc
// RUN: llvm-as %t/unresolved.ll -o %t/unresolved.bc
// RUN: llvm-as %t/two-kernels.ll -o %t/two-kernels.bc
// RUN: chmod +x %t/ocloc

// The device images of every IGCA target are processed separately and the
// images which define the same kernels are concatenated into an uber-module.
// Undefined symbols are errors because the device code is fully linked.
// RUN: llvm-offload-binary -o %t/multi.pack \
// RUN:   --image=file=%t/resolved.bc,triple=spir64-unknown-unknown,arch=igca_30r,kind=sycl \
// RUN:   --image=file=%t/resolved.bc,triple=spir64-unknown-unknown,arch=igca_20ca,kind=sycl,code=xe-pvc
// RUN: clang-linker-wrapper --dry-run --no-sycl-rdc --ocloc-path=%t \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld \
// RUN:   %t/multi.pack -o %t/a.out 2>&1 | FileCheck %s --check-prefix=WRAP
// WRAP-DAG: sycl-post-link{{.*}} -undefined-symbols-are-errors {{.*}}igca_20ca{{.*}}.bc
// WRAP-DAG: sycl-post-link{{.*}} -undefined-symbols-are-errors {{.*}}igca_30r{{.*}}.bc
// WRAP-DAG: llvm-spirv{{.*}} -o [[SPV0:.*\.spv]]
// WRAP-DAG: llvm-spirv{{.*}} -o [[SPV1:.*\.spv]]
// WRAP: "{{.*}}ocloc" concat {{.*}}.spv {{.*}}.spv -out [[UBER:.*\.sycl\.uber.*\.bin]]
// WRAP-NOT: ocloc
// WRAP-NEXT: offload-wrapper: output: [[WRAPPER:.*\.bc]], input: [[UBER]]
// WRAP-NEXT: clang"{{.*}} -c -o [[OBJ:.*\.o]] [[WRAPPER]]
// WRAP-NEXT: "/usr/bin/ld" {{.*}}[[OBJ]]

// With --emit-fatbin-only the uber-modules are stored in the output container.
// RUN: clang-linker-wrapper --dry-run --no-sycl-rdc --emit-fatbin-only \
// RUN:   --ocloc-path=%t --linker-path=/usr/bin/ld \
// RUN:   %t/multi.pack -o %t/multi.fatbin 2>&1 | FileCheck %s --check-prefix=FATBIN
// FATBIN: "{{.*}}ocloc" concat {{.*}}.spv {{.*}}.spv -out {{.*}}.sycl.uber{{.*}}.bin
// FATBIN-NOT: offload-wrapper
// FATBIN-NOT: "/usr/bin/ld"
// RUN: llvm-objdump --offloading %t/multi.fatbin | FileCheck %s --check-prefix=CONTAINER
// CONTAINER: OFFLOADING IMAGE [0]:
// CONTAINER: kind elf
// CONTAINER: arch igca_20ca,igca_30r
// CONTAINER: triple spir64-unknown-unknown
// CONTAINER: producer sycl
// CONTAINER-NOT: OFFLOADING IMAGE

// A single IGCA target does not require ocloc.
// RUN: llvm-offload-binary -o %t/single.pack \
// RUN:   --image=file=%t/resolved.bc,triple=spir64-unknown-unknown,arch=igca_20ca,kind=sycl
// RUN: clang-linker-wrapper --dry-run --no-sycl-rdc --emit-fatbin-only \
// RUN:   --linker-path=/usr/bin/ld %t/single.pack -o %t/single.fatbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SINGLE
// SINGLE: llvm-spirv
// SINGLE-NOT: ocloc

// Undefined symbols are diagnosed by the SYCL post-link step.
// RUN: llvm-offload-binary -o %t/unresolved.pack \
// RUN:   --image=file=%t/unresolved.bc,triple=spir64-unknown-unknown,arch=igca_20ca,kind=sycl
// RUN: not clang-linker-wrapper --no-sycl-rdc --emit-fatbin-only \
// RUN:   --linker-path=/usr/bin/ld %t/unresolved.pack -o %t/unresolved.fatbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNRESOLVED
// RUN: not clang-linker-wrapper --no-sycl-rdc --emit-fatbin-only \
// RUN:   --no-use-sycl-post-link-tool --linker-path=/usr/bin/ld \
// RUN:   %t/unresolved.pack -o %t/unresolved.fatbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNRESOLVED
// UNRESOLVED-NOT: warning: Undefined function
// UNRESOLVED: error: {{.*}}undefined symbol(s) _Z3foov, _Z3bar found in

// Undefined symbols are allowed with dynamic linking of device images.
// RUN: clang-linker-wrapper --no-sycl-rdc --emit-fatbin-only \
// RUN:   --sycl-allow-device-image-dependencies \
// RUN:   --sycl-post-link-options=-allow-device-image-dependencies \
// RUN:   --linker-path=/usr/bin/ld %t/unresolved.pack -o %t/deps.fatbin
// RUN: llvm-objdump --offloading %t/deps.fatbin | FileCheck %s --check-prefix=DEPS
// DEPS: arch igca_20ca
// RUN: clang-linker-wrapper --dry-run --no-sycl-rdc --emit-fatbin-only \
// RUN:   --sycl-allow-device-image-dependencies --linker-path=/usr/bin/ld \
// RUN:   %t/unresolved.pack -o %t/deps.fatbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-ERRORS
// NO-ERRORS-NOT: -undefined-symbols-are-errors

// With per-kernel split, every uber-module holds the images of one kernel.
// The stub ocloc concatenates its inputs.
// RUN: llvm-offload-binary -o %t/split.pack \
// RUN:   --image=file=%t/two-kernels.bc,triple=spir64-unknown-unknown,arch=igca_20ca,kind=sycl \
// RUN:   --image=file=%t/two-kernels.bc,triple=spir64-unknown-unknown,arch=igca_30r,kind=sycl
// RUN: clang-linker-wrapper --no-sycl-rdc --emit-fatbin-only --ocloc-path=%t \
// RUN:   --sycl-post-link-options=-split=kernel --linker-path=/usr/bin/ld \
// RUN:   %t/split.pack -o %t/split.fatbin 2>&1 | FileCheck %s --check-prefix=SPLIT
// SPLIT-COUNT-2: ocloc-stub: concat {{.*}}.spv {{.*}}.spv -out {{.*}}.bin
// SPLIT-NOT: ocloc-stub
// RUN: llvm-objdump --offloading %t/split.fatbin | FileCheck %s --check-prefix=SPLIT-CONTAINER
// SPLIT-CONTAINER-COUNT-2: arch igca_20ca,igca_30r
// SPLIT-CONTAINER-NOT: OFFLOADING IMAGE

// IGCA targets which define different kernels are diagnosed.
// RUN: llvm-offload-binary -o %t/mismatch.pack \
// RUN:   --image=file=%t/two-kernels.bc,triple=spir64-unknown-unknown,arch=igca_20ca,kind=sycl \
// RUN:   --image=file=%t/resolved.bc,triple=spir64-unknown-unknown,arch=igca_30r,kind=sycl
// RUN: not clang-linker-wrapper --no-sycl-rdc --emit-fatbin-only --ocloc-path=%t \
// RUN:   --linker-path=/usr/bin/ld %t/mismatch.pack -o %t/mismatch.fatbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISMATCH
// MISMATCH: error: device images of IGCA target 'igca_30r' do not define the same kernels as other IGCA targets

//--- resolved.ll
target triple = "spir64-unknown-unknown"

define spir_kernel void @kernel() {
  ret void
}

//--- unresolved.ll
target triple = "spir64-unknown-unknown"

declare spir_func void @_Z3foov()
declare spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
declare void @llvm.assume(i1)

@_Z3bar = external addrspace(1) global i32
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>

define spir_kernel void @kernel() {
  call spir_func void @_Z3foov()
  %id = call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  call void @llvm.assume(i1 true)
  %val = load i32, ptr addrspace(1) @_Z3bar
  %gid = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  ret void
}

//--- two-kernels.ll
target triple = "spir64-unknown-unknown"

define spir_kernel void @kernel1() #0 {
  ret void
}

define spir_kernel void @kernel2() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="two-kernels.cpp" }

//--- ocloc
#!/bin/sh
echo "ocloc-stub: $*" >&2
shift
inputs=""
while [ "$1" != "-out" ]; do inputs="$inputs $1"; shift; done
cat $inputs > "$2"
