/// Check "-aux-target-cpu" and "target-cpu" are passed when compiling for SYCL offload device and host codes:
//  RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OFFLOAD %s
//  CHECK-OFFLOAD: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  CHECK-OFFLOAD-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]"
//  CHECK-OFFLOAD-NEXT: append-file{{.*}}
//  CHECK-OFFLOAD-NEXT: clang{{.*}} "-cc1" {{.*}}
//  CHECK-OFFLOAD-NEXT-SAME: "-fsycl-is-host"
//  CHECK-OFFLOAD-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]"

/// Check "-aux-target-cpu" with "-aux-target-feature" and "-target-cpu" with "-target-feature" are passed
/// when compiling for SYCL offload device and host codes:
//  RUN:  %clang -fsycl -mavx -c %s -### -o %t.o 2>&1 | FileCheck -check-prefix=OFFLOAD-AVX %s
//  OFFLOAD-AVX: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  OFFLOAD-AVX-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]" "-aux-target-feature" "+avx"
//  OFFLOAD-AVX-NEXT: append-file{{.*}}
//  OFFLOAD-AVX-NEXT: clang{{.*}} "-cc1" {{.*}}
//  OFFLOAD-AVX-NEXT-SAME: "-fsycl-is-host"
//  OFFLOAD-AVX-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]" "-target-feature" "+avx"

/// Check that the needed -fsycl -fsycl-is-device and -fsycl-is-host options
/// are passed to all of the needed compilation steps regardless of final
/// phase.
// RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl -E %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl -S %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"

/// Check that -fcoverage-mapping is disabled for device
// RUN: %clang -### -fsycl -fprofile-instr-generate -fcoverage-mapping -target x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_COVERAGE_MAPPING %s
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fprofile-instrument=clang"
// CHECK_COVERAGE_MAPPING-NOT: "-fcoverage-mapping"
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-fprofile-instrument=clang"{{.*}} "-fcoverage-mapping"{{.*}}

/// check for PIC for device wrap compilation when using -shared or -fPIC
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -shared %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SHARED %s
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fPIC %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SHARED %s
// CHECK_SHARED: llc{{.*}} "-relocation-model=pic"

/// -S -emit-llvm should generate textual IR for device.
// RUN: %clangxx -### -fsycl -S -emit-llvm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_S_LLVM %s
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm"{{.*}} "-o" "[[DEVICE:.+\.ll]]"
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-host"{{.*}} "-emit-llvm"{{.*}} "-o" "[[HOST:.+\.ll]]"
// CHECK_S_LLVM: clang-offload-bundler{{.*}} "-type=ll"{{.*}} "-inputs=[[DEVICE]],[[HOST]]"

/// Check for default device triple compilations based on object, archive or
/// forced from command line.
// RUN:  touch %t_empty.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// IMPLIED_DEVICE_OBJ: clang-offload-bundler{{.*}} "-type=o"{{.*}} "-targets={{.*}}sycl-spir64-unknown-unknown-sycldevice,sycl-spir64_{{.*}}-unknown-unknown-sycldevice"{{.*}} "-unbundle"

// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// IMPLIED_DEVICE_LIB: clang-offload-bundler{{.*}} "-type=a"{{.*}} "-targets=sycl-spir64-unknown-unknown-sycldevice,sycl-spir64_{{.*}}-unknown-unknown-sycldevice"{{.*}} "-unbundle"

/// Check that the default device triple is not used with -fno-sycl-link-spirv
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_CPU %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_fpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_FPGA %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_GEN %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fintelfpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_FPGA %s
// NO_IMPLIED_DEVICE_CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown-sycldevice"
// NO_IMPLIED_DEVICE_FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"
// NO_IMPLIED_DEVICE_GEN: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice"
// NO_IMPLIED_DEVICE_OPT-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice"{{.*}} "-check-section"
// NO_IMPLIED_DEVICE_OPT-NOT: clang-offload-bundler{{.*}} "-targets={{.*}}spir64-unknown-unknown-sycldevice{{.*}}" "-unbundle"

// RUN:  %clangxx -### -fsycl -fsycl-targets=spir64_x86_64 %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -fsycl -fsycl-targets=spir64_fpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -fsycl -fsycl-targets=spir64_gen %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -fsycl -fintelfpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// NO_IMPLIED_DEVICE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice"{{.*}} "-check-section"
// NO_IMPLIED_DEVICE-NOT: clang-offload-bundler{{.*}} "-targets={{.*}}spir64-unknown-unknown-sycldevice{{.*}}" "-unbundle"

/// Passing in the default triple should allow for -Xsycl-target options
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// SYCL_TARGET_OPT: clang-offload-wrapper{{.*}} "-compile-opts=-DFOO" "-link-opts=-DFOO2"
