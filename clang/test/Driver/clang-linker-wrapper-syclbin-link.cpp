/// Check that clang-linker-wrapper can link SYCLBIN files in input or object
/// state into a single SYCLBIN file in executable state.

// UNSUPPORTED: system-windows

// Two SYCLBIN files where one uses a SYCL_EXTERNAL function defined by the
// other. 'a' is in input state and 'b' is in object state, to cover both of the
// linkable bundle states.
//
// RUN: %clang -fsycl -fsyclbin=input --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code %S/Inputs/syclbin-link-a.cpp -o %t.a.syclbin
// RUN: %clang -fsycl -fsyclbin=object --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code %S/Inputs/syclbin-link-b.cpp -o %t.b.syclbin
// RUN: %clang -fsycl -fsyclbin=executable --offload-new-driver --no-offloadlib \
// RUN:   -fno-sycl-instrument-device-code %S/Inputs/syclbin-link-b.cpp -o %t.b.exe.syclbin

/// The device images held by the SYCLBIN files are extracted and go through the
/// regular SYCL device linking pipeline, and the result is packaged into a
/// SYCLBIN file that is copied to the output.  No host code is involved.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable -o %t.out.syclbin \
// RUN:   %t.a.syclbin %t.b.syclbin --dry-run 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-CMDS
// CHK-CMDS:      spirv-to-ir-wrapper{{.*}} -o [[FIRSTLINKIN:.*]].bc
// CHK-CMDS-NEXT: spirv-to-ir-wrapper{{.*}} -o [[SECONDLINKIN:.*]].bc
// CHK-CMDS-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLINKIN]].bc [[SECONDLINKIN]].bc -o [[LINKOUT:.*]].bc
// CHK-CMDS-NEXT: sycl-post-link{{.*}} -o {{.*}}.table [[LINKOUT]].bc
// CHK-CMDS-NEXT: llvm-spirv{{.*}} -o {{.*}}
// CHK-CMDS-NEXT: "{{.*cp|copy}}" {{.*}}.syclbin {{.*}}.out.syclbin
// CHK-CMDS-NOT:  "{{.*}}/ld"

/// With '--syclbin-link-target' the linked device code is compiled ahead of
/// time for the named target instead of being left for the runtime.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable \
// RUN:   --syclbin-link-target=spir64_gen-unknown-unknown=pvc \
// RUN:   -o %t.aot.syclbin %t.a.syclbin %t.b.syclbin --dry-run 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-AOT
// CHK-AOT:      spirv-to-ir-wrapper{{.*}} -o [[FIRSTLINKIN:.*]].bc
// CHK-AOT-NEXT: spirv-to-ir-wrapper{{.*}} -o [[SECONDLINKIN:.*]].bc
// CHK-AOT-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLINKIN]].bc [[SECONDLINKIN]].bc -o [[LINKOUT:.*]].bc
// CHK-AOT-NEXT: sycl-post-link{{.*}} -o intel_gpu_pvc,{{.*}}.table [[LINKOUT]].bc
// CHK-AOT-NEXT: llvm-spirv{{.*}} -o [[SPVOUT:.*]].spv {{.*}}
// CHK-AOT-NEXT: "{{.*}}ocloc" -output_no_suffix -spirv_input -device pvc -output {{.*}} -file [[SPVOUT]].spv
// CHK-AOT-NEXT: "{{.*cp|copy}}" {{.*}}.syclbin {{.*}}.aot.syclbin

/// Every requested target is compiled for separately, and the results end up in
/// the same SYCLBIN file.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable \
// RUN:   --syclbin-link-target=spir64_gen-unknown-unknown=pvc \
// RUN:   --syclbin-link-target=spir64_gen-unknown-unknown=bmg_g21 \
// RUN:   -o %t.aot2.syclbin %t.a.syclbin %t.b.syclbin --dry-run 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-AOT-MULTI
// CHK-AOT-MULTI:      "{{.*}}ocloc" -output_no_suffix -spirv_input -device pvc
// CHK-AOT-MULTI:      "{{.*}}ocloc" -output_no_suffix -spirv_input -device bmg_g21
// CHK-AOT-MULTI-NEXT: "{{.*cp|copy}}" {{.*}}.syclbin {{.*}}.aot2.syclbin

// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable \
// RUN:   --syclbin-link-target=pvc -o %t.err.syclbin \
// RUN:   %t.a.syclbin %t.b.syclbin --dry-run 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-BAD-TARGET
// CHK-BAD-TARGET: error: expected '<triple>=<arch>' in '--syclbin-link-target=pvc'

/// A target that none of the inputs holds device code for would silently be
/// missing from the output, so it is reported instead.
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable \
// RUN:   --syclbin-link-target=nvptx64-nvidia-cuda=sm_80 -o %t.err.syclbin \
// RUN:   %t.a.syclbin %t.b.syclbin --dry-run 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-UNCOVERED-TARGET
// CHK-UNCOVERED-TARGET: error: none of the SYCLBIN files being linked contains device code that can be compiled for 'sm_80' (nvptx64-nvidia-cuda)

/// The output is a SYCLBIN file in executable state, so it cannot be linked
/// again.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable -o %t.out.syclbin \
// RUN:   %t.a.syclbin %t.b.syclbin
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable -o %t.err.syclbin \
// RUN:   %t.out.syclbin 2>&1 | FileCheck %s --check-prefix=CHK-EXECUTABLE-INPUT
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable -o %t.err.syclbin \
// RUN:   %t.a.syclbin %t.b.exe.syclbin 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK-EXECUTABLE-INPUT
// CHK-EXECUTABLE-INPUT: error: SYCLBIN file '{{.*}}.syclbin' is in executable state; only SYCLBIN files in input or object state can be linked

/// A SYCL_EXTERNAL function that is used but not defined by any of the inputs
/// cannot be resolved anymore once the output is in executable state, so it is
/// an error rather than a warning.
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable -o %t.err.syclbin \
// RUN:   %t.a.syclbin 2>&1 | FileCheck %s --check-prefix=CHK-UNDEFINED
// CHK-UNDEFINED: error: undefined SYCL_EXTERNAL function in the device code being linked:
// CHK-UNDEFINED-NEXT: syclbin_link_foo(int)
// CHK-UNDEFINED-NEXT: provide the definition in one of the linked inputs

/// Undefined symbols are expected to be resolved from other device images at
/// run time when device image dependencies are allowed, so they are not
/// diagnosed then.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --syclbin=executable \
// RUN:   --sycl-allow-device-image-dependencies -o %t.dep.syclbin \
// RUN:   %t.a.syclbin
