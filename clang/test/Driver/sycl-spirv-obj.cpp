///
/// Tests for SPIR-V related objects
///

/// -fsycl-device-obj=spirv
// RUN: %clangxx -target x86_64-unknown-linux-gnu -c -fsycl --offload-new-driver -fsycl-device-obj=spirv -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix SPIRV_DEVICE_OBJ
// SPIRV_DEVICE_OBJ: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"
// SPIRV_DEVICE_OBJ-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// SPIRV_DEVICE_OBJ-SAME: "-fsycl-is-device"
// SPIRV_DEVICE_OBJ-SAME: "-o" "[[DEVICE_BC:.+\.bc]]"
// SPIRV_DEVICE_OBJ: llvm-spirv{{.*}} "-o" "[[DEVICE_SPV:.+\.spv]]"
// SPIRV_DEVICE_OBJ-SAME: "--spirv-preserve-auxdata"
// SPIRV_DEVICE_OBJ-SAME: "-spirv-ext=-all,{{.*}},+SPV_KHR_cooperative_matrix"
// SPIRV_DEVICE_OBJ-SAME: "[[DEVICE_BC]]"
// SPIRV_DEVICE_OBJ: clang-offload-packager{{.*}} "--image=file=[[DEVICE_SPV]]{{.*}}"
// SPIRV_DEVICE_OBJ: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// SPIRV_DEVICE_OBJ-SAME: "-fsycl-is-host"
// SPIRV_DEVICE_OBJ-SAME: "-o" "[[HOST_OBJ:.+\.o]]"

// RUN: %clangxx -target x86_64-unknown-linux-gnu -c -fsycl --offload-new-driver -fsycl-device-obj=spirv -ccc-print-phases %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix SPIRV_DEVICE_OBJ_PHASES
// SPIRV_DEVICE_OBJ_PHASES: 0: input, "[[INPUTSRC:.+\.cpp]]", c++, (host-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 1: append-footer, {0}, c++, (host-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 3: compiler, {2}, ir, (host-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 4: input, "[[INPUTSRC]]", c++, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 6: compiler, {5}, ir, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 7: backend, {6}, ir, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 8: llvm-spirv, {7}, spirv, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 9: offload, "device-sycl (spir64-unknown-unknown)" {8}, spirv
// SPIRV_DEVICE_OBJ_PHASES: 10: clang-offload-packager, {9}, image, (device-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {10}, ir
// SPIRV_DEVICE_OBJ_PHASES: 12: backend, {11}, assembler, (host-sycl)
// SPIRV_DEVICE_OBJ_PHASES: 13: assembler, {12}, object, (host-sycl)

/// Use of -fsycl-device-obj=spirv should not be effective during linking
// RUN: touch %t.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-device-obj=spirv -### %t.o 2>&1 | \
// RUN:  FileCheck %s -check-prefixes=OPT_WARNING,LLVM_SPIRV_R
// OPT_WARNING: warning: argument unused during compilation: '-fsycl-device-obj=spirv'
// LLVM_SPIRV_R: clang-linker-wrapper{{.*}}
