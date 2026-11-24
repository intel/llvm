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
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.mixed.ii -output=%t.res1 -output=%t.res2 -output=%t.res3 -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-pvc,sycl-spir64_gen-unknown-unknown-acm_g10,sycl-spir64_gen-unknown-unknown
// RUN: diff %t.tgt2 %t.res1
// RUN: diff %t.noid %t.res2
// RUN: diff %t.noid %t.res3

// The entry without a target ID is used for each of several target IDs, as
// for -fsycl-targets=intel_gpu_tgllp,intel_gpu_pvc.
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.noid.ii -output=%t.res1 -output=%t.res2 -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt1 %t.res1
// RUN: diff %t.tgt1 %t.res2
// RUN: clang-offload-bundler -type=o -input=%t.o -input=%t.tgt1 -output=%t.bundle.noid.o \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown
// RUN: clang-offload-bundler -type=o -input=%t.bundle.noid.o -output=%t.res1 -output=%t.res2 -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-pvc
// RUN: diff %t.tgt1 %t.res1
// RUN: diff %t.tgt1 %t.res2

// The entry without a target ID is used for each target ID, including Intel
// GPUs that are not known as an offload arch. For spir64_gen, the last
// component is always the target ID, not the triple environment; otherwise
// only one of those targets would get the code.
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.noid.ii -unbundle \
// RUN:   -output=%t.new1 -output=%t.new2 -output=%t.new3 -output=%t.new4 \
// RUN:   -output=%t.new5 -output=%t.new6 -output=%t.new7 -output=%t.new8 \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-bmg_g31,sycl-spir64_gen-unknown-unknown-ptl_h,sycl-spir64_gen-unknown-unknown-ptl_u,sycl-spir64_gen-unknown-unknown-wcl,sycl-spir64_gen-unknown-unknown-nvl_s,sycl-spir64_gen-unknown-unknown-nvl_u,sycl-spir64_gen-unknown-unknown-nvl_p,sycl-spir64_gen-unknown-unknown-cri
// RUN: diff %t.tgt1 %t.new1
// RUN: diff %t.tgt1 %t.new2
// RUN: diff %t.tgt1 %t.new3
// RUN: diff %t.tgt1 %t.new4
// RUN: diff %t.tgt1 %t.new5
// RUN: diff %t.tgt1 %t.new6
// RUN: diff %t.tgt1 %t.new7
// RUN: diff %t.tgt1 %t.new8

// The same for an archive, as for a static library compiled with
// -fsycl-targets=spir64_gen and linked for many intel_gpu_* targets.
// RUN: rm -f %t.noid.a
// RUN: llvm-ar crs %t.noid.a %t.bundle.noid.o
// RUN: clang-offload-bundler -type=ao -input=%t.noid.a -unbundle -allow-missing-bundles \
// RUN:   -output=%t.ar1 -output=%t.ar2 -output=%t.ar3 -output=%t.ar4 -output=%t.ar5 \
// RUN:   -output=%t.ar6 -output=%t.ar7 -output=%t.ar8 -output=%t.ar9 -output=%t.ar10 \
// RUN:   -output=%t.ar11 -output=%t.ar12 -output=%t.ar13 -output=%t.ar14 \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-tgllp,sycl-spir64_gen-unknown-unknown-adl_s,sycl-spir64_gen-unknown-unknown-adl_p,sycl-spir64_gen-unknown-unknown-adl_n,sycl-spir64_gen-unknown-unknown-dg2,sycl-spir64_gen-unknown-unknown-pvc,sycl-spir64_gen-unknown-unknown-mtl,sycl-spir64_gen-unknown-unknown-mtl_u,sycl-spir64_gen-unknown-unknown-arl_h,sycl-spir64_gen-unknown-unknown-bmg,sycl-spir64_gen-unknown-unknown-lnl_m,sycl-spir64_gen-unknown-unknown-ptl_h,sycl-spir64_gen-unknown-unknown-ptl_u,sycl-spir64_gen-unknown-unknown-nvl_s
// RUN: diff %t.tgt1 %t.ar1
// RUN: diff %t.tgt1 %t.ar2
// RUN: diff %t.tgt1 %t.ar3
// RUN: diff %t.tgt1 %t.ar4
// RUN: diff %t.tgt1 %t.ar5
// RUN: diff %t.tgt1 %t.ar6
// RUN: diff %t.tgt1 %t.ar7
// RUN: diff %t.tgt1 %t.ar8
// RUN: diff %t.tgt1 %t.ar9
// RUN: diff %t.tgt1 %t.ar10
// RUN: diff %t.tgt1 %t.ar11
// RUN: diff %t.tgt1 %t.ar12
// RUN: diff %t.tgt1 %t.ar13
// RUN: diff %t.tgt1 %t.ar14

// For other SYCL triples, the last component is still not treated as the
// target ID unless it is a known offload arch. For
// sycl-spir64-unknown-unknown-foo, foo is the (unknown) triple environment, so
// there is no target ID to mismatch with bar. For spir64_gen, foo and bar are
// different target IDs.
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -output=%t.bundle.env.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown-foo
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.env.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64-unknown-unknown-bar
// RUN: diff %t.tgt1 %t.res
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -output=%t.bundle.gen.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-foo
// RUN: not clang-offload-bundler -type=ii -input=%t.bundle.gen.ii -output=%t.res -unbundle \
// RUN:   -targets=sycl-spir64_gen-unknown-unknown-bar 2>&1 | FileCheck %s --check-prefix=GEN-MISMATCH
// GEN-MISMATCH: error: Can't find bundles for sycl-spir64_gen-unknown-unknown-bar

// 4-component triples are unaffected.
// RUN: clang-offload-bundler -type=ii -input=%t.host -input=%t.tgt1 -input=%t.tgt2 -output=%t.bundle.hip.ii \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx908,hipv4-amdgcn-amd-amdhsa--gfx90a
// RUN: clang-offload-bundler -type=ii -input=%t.bundle.hip.ii -output=%t.res -unbundle \
// RUN:   -targets=hipv4-amdgcn-amd-amdhsa--gfx90a
// RUN: diff %t.tgt2 %t.res

int A = 0;
