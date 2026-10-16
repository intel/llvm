// REQUIRES: x86-registered-target

// SYCL bundle entry IDs use a 3-component triple followed by the target ID,
// e.g. sycl-spir64_gen-unknown-unknown-pvc. Check that each target ID is
// unbundled to its own code object.

// RUN: %clang -O0 -target x86_64-unknown-linux-gnu %s -c -o %t.o
// RUN: echo 'host' > %t.host
// RUN: echo 'tgt1' > %t.tgt1
// RUN: echo 'tgt2' > %t.tgt2

// Text bundles.
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.ii -output=%t.res1 -output=%t.res2 -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt1 %t.res1
// RUN: diff %t.tgt2 %t.res2
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt2 %t.res

// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.nv.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-nvptx64-nvidia-cuda-sm_80,sycl-nvptx64-nvidia-cuda-sm_90
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.nv.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-nvptx64-nvidia-cuda-sm_90
// RUN: diff %t.tgt2 %t.res

// Object bundles, as produced for -fsycl-targets=intel_gpu_* with and without
// -fno-sycl-rdc. With -fno-sycl-rdc, the old offloading model names the SPIR
// AOT triple spir64_gen_image.
// RUN: clang-offload-bundler -type=o -input=%t.o -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.o \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: clang-offload-bundler -type=o -input=%t.bundle.o -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt2 %t.res
// RUN: clang-offload-bundler -type=o -input=%t.o -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.image.o \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen_image-unknown-unknown-tgllp,sycl-spir64_gen_image-unknown-unknown-pvc
// RUN: clang-offload-bundler -type=o -input=%t.bundle.image.o -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen_image-unknown-unknown-pvc
// RUN: diff %t.tgt2 %t.res

// A SYCL code object without a target ID is still compatible with any target
// ID of the same triple, and vice versa.
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -output=%t.bundle.noid.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.noid.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt1 %t.res
// With no exact match for a target without a target ID, the first compatible
// entry is used (tgllp here, as it comes before pvc in the bundle).
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown
// RUN: diff %t.tgt1 %t.res

// With entries both with and without a target ID, an exact target ID match is
// used regardless of the order of the entries.
// RUN: echo 'noid' > %t.noid
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.noid -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.mixed.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown,sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.mixed.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt2 %t.res
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.mixed.ii -output=%t.res1 -output=%t.res2 -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown
// RUN: diff %t.tgt1 %t.res1
// RUN: diff %t.noid %t.res2
// The entry without a target ID is used for a target ID with no exact entry.
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.mixed.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-acm_g10
// RUN: diff %t.noid %t.res

// 4-component triples are unaffected.
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.hip.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx908,hipv4-amdgcn-amd-amdhsa--gfx90a
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.hip.ii -output=%t.res -unbundle \
// RUN:   -targets=hipv4-amdgcn-amd-amdhsa--gfx90a
// RUN: diff %t.tgt2 %t.res

int A = 0;
