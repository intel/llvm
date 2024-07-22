///
/// Perform several driver tests for SYCL offloading with AOT enabled
///


// Check that when -fintelfpga is passed without -fsycl, no error is thrown.
// RUN:   %clang -### -fintelfpga  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-FINTELFPGA %s
// CHK-NO-FSYCL-FINTELFPGA-NOT: error: '-fintelfpga' must be used in conjunction with '-fsycl' to enable offloading

/// Check error for -fsycl-targets -fintelfpga conflict
// RUN:   not %clang -### -fsycl-targets=spir64-unknown-unknown -fintelfpga  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// RUN:   not %clang_cl -### -fsycl-targets=spir64-unknown-unknown -fintelfpga  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// CHK-SYCL-FPGA-CONFLICT: clang: error: the option -fsycl-targets= conflicts with -fintelfpga

/// Check that -aux-triple is passed with -fintelfpga
// RUN:    %clang -### -fintelfpga %s 2>&1 \
// RUN:    | FileCheck -DARCH=spir64_fpga -check-prefix=CHK-SYCL-FPGA-AUX-TRIPLE %s
// CHK-SYCL-FPGA-AUX-TRIPLE: clang{{.*}} "-cc1" "-triple" "{{.*}}"{{.*}} "-aux-triple" "[[ARCH]]-{{.*}}"{{.*}} "-fsycl-is-host"

/// Check error for -fsycl-targets with bad triple
// RUN:   not %clang -### -fsycl-targets=spir64_bad-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-TRIPLE %s
// RUN:   not %clang_cl -### -fsycl-targets=spir64_bad-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-TRIPLE %s
// CHK-SYCL-BAD-TRIPLE: error: SYCL target is invalid: 'spir64_bad-unknown-unknown'

/// Check no error for -fsycl-targets with good triple
// RUN:   %clang -### -fsycl-targets=spir64_fpga-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_fpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_x86_64-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_x86_64 -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_gen-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_gen -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// CHK-SYCL-TARGET-NOT: error: SYCL target is invalid

/// Check -Xsycl-target-frontend= accepts triple aliases
// RUN:   %clang -### -fsycl -fsycl-targets=spir64_x86_64 -Xsycl-target-frontend=spir64_x86_64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH=spir64_x86_64 -check-prefixes=CHK-UNUSED-ARG-WARNING,CHK-TARGET %s
// RUN:   %clang -### -fsycl -fsycl-targets=spir64_gen -Xsycl-target-frontend=spir64_gen -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH=spir64_gen -check-prefixes=CHK-UNUSED-ARG-WARNING,CHK-TARGET %s
// RUN:   %clang -### -fsycl -fsycl-targets=spir64_fpga -Xsycl-target-frontend=spir64_fpga -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH=spir64_fpga -check-prefixes=CHK-UNUSED-ARG-WARNING,CHK-TARGET %s
// CHK-UNUSED-ARG-WARNING-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET: clang{{.*}} "-cc1" "-triple" "[[ARCH]]-unknown-unknown"{{.*}} "-D" "FOO"

/// Check -Xsycl-target-backend triggers error when multiple triples are used.
// RUN:   not %clang -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown,spir_fpga-unknown-unknown -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// RUN:   not %clang_cl -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown,spir_fpga-unknown-unknown -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// CHK-FSYCL-TARGET-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-backend', specify triple using '-Xsycl-target-backend=<triple>'

/// Check -Xsycl-target-* does not trigger an error when multiple instances of
/// -fsycl-targets is used.
// RUN:   %clang -### -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-2X-ERROR %s
// RUN:   %clang -### -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-2X-ERROR %s
// RUN:   %clang -### -fsycl -fsycl-targets=spir64-unknown-unknown -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-linker -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-2X-ERROR %s
// CHK-FSYCL-TARGET-2X-ERROR-NOT: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target{{.*}}', specify triple using '-Xsycl-target{{.*}}=<triple>'

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_gen %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64 %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASES-AOT: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-AOT: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-AOT: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES-AOT: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-FPGA: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-GEN: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-CPU: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_x86_64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-AOT: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-AOT: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-AOT: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-AOT: 9: linker, {4}, ir, (device-sycl)
// CHK-PHASES-AOT: 10: sycl-post-link, {9}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 11: file-table-tform, {10}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 12: llvm-spirv, {11}, tempfilelist, (device-sycl)
// CHK-PHASES-FPGA: 13: backend-compiler, {12}, fpga_aocx, (device-sycl)
// CHK-PHASES-CPU: 13: backend-compiler, {12}, image, (device-sycl)
// CHK-PHASES-GEN: 13: backend-compiler, {12}, image, (device-sycl)
// CHK-PHASES-AOT: 14: file-table-tform, {10, 13}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES-FPGA: 16: offload, "device-sycl (spir64_fpga-unknown-unknown)" {15}, object
// CHK-PHASES-CPU: 16: offload, "device-sycl (spir64_x86_64-unknown-unknown)" {15}, object
// CHK-PHASES-GEN: 16: offload, "device-sycl (spir64_gen-unknown-unknown)" {15}, object
// CHK-PHASES-AOT: 17: linker, {8, 16}, image, (host-sycl)

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fintelfpga -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown"
// CHK-TOOLS-GEN: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// CHK-TOOLS-CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown"
// CHK-TOOLS-AOT: "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT7:.+\.o]]
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "-o" "[[OUTPUT2_T:.+\.table]]" "[[OUTPUT2]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-extract=Code" "-drop_titles" "-o" "[[OUTPUT2_1:.+\.txt]]" "[[OUTPUT2_T]]"
// CHK-TOOLS-CPU: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.4" "-spirv-debug-info-version=nonsemantic-shader-200" "-spirv-allow-unknown-intrinsics=llvm.genx.,llvm.fpbuiltin" {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-GEN: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.4" "-spirv-debug-info-version=nonsemantic-shader-200" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-HW: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.4" "-spirv-debug-info-version=ocl-100" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-EMU: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.4" "-spirv-debug-info-version=nonsemantic-shader-200" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-HW: aoc{{.*}} "-o" "[[OUTPUT4_T:.+\.aocx]]" "[[OUTPUT3_T]]"
// CHK-TOOLS-FPGA-EMU: opencl-aot{{.*}} "-spv=[[OUTPUT3_T]]" "-ir=[[OUTPUT4_T:.+\.aocx]]"
// CHK-TOOLS-GEN: ocloc{{.*}} "-output" "[[OUTPUT4_T:.+\.out]]" {{.*}} "[[OUTPUT3_T]]"
// CHK-TOOLS-CPU: opencl-aot{{.*}} "-o=[[OUTPUT4_T:.+\.out]]" {{.*}} "[[OUTPUT3_T]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-o" "[[OUTPUT4:.+\.table]]" "{{.*}}.table"{{.*}} "[[OUTPUT4_T]]"
// CHK-TOOLS-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-GEN: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_gen{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-CPU: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_x86_64{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-AOT: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT6:.+\.o]]" "[[OUTPUT5]]"
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT7]]" "[[OUTPUT6]]" {{.*}} "-lsycl"

// Check to be sure that for windows, the 'exe' tools are called
// RUN: %clang_cl -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-CPU-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-CPU-WIN
// RUN: %clang_cl -fsycl -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-GEN-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-GEN-WIN
// RUN: %clang_cl -fsycl -Xshardware -fsycl-targets=spir64_fpga-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-FPGA-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-FPGA-WIN
// CHK-TOOLS-GEN-WIN: ocloc.exe{{.*}}
// CHK-TOOLS-CPU-WIN: opencl-aot.exe{{.*}}
// CHK-TOOLS-FPGA-WIN: aoc.exe{{.*}}

/// ###########################################################################

/// Check -Xsycl-target-backend option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
/// Check -Xs option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fintelfpga -XsDFOO1 -XsDFOO2 -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fintelfpga -Xs "-DFOO1 -DFOO2" -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// CHK-TOOLS-FPGA-OPTS: aoc{{.*}} "-o" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-FPGA-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-GEN-OPTS %s
// CHK-TOOLS-GEN-OPTS: ocloc{{.*}} "-output" {{.*}} "-output_no_suffix" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-GEN-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS %s
// CHK-TOOLS-CPU-OPTS: opencl-aot{{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-CPU-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -Xsycl-target-backend "--bo='\"-DFOO1 -DFOO2\"'" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS3 %s
// CHK-TOOLS-CPU-OPTS3: opencl-aot{{.*}} "--bo=\"-DFOO1 -DFOO2\""

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-FPGA %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-FPGA %s
// CHK-TOOLS-IMPLIED-OPTS-FPGA: opencl-aot{{.*}} "--bo=-g -cl-opt-disable" "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-CPU %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-CPU %s
// CHK-TOOLS-IMPLIED-OPTS-CPU: opencl-aot{{.*}} "--bo=-g -cl-opt-disable" "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-GEN %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_gen-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-GEN %s
// CHK-TOOLS-IMPLIED-OPTS-GEN: ocloc{{.*}} "-options" "-g -cl-opt-disable" "-DFOO1" "-DFOO2"

/// Check -Xsycl-target-linker option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS2 %s
// CHK-TOOLS-FPGA-OPTS2: aoc{{.*}} "-o" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-FPGA-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-GEN-OPTS2 %s
// CHK-TOOLS-GEN-OPTS2: ocloc{{.*}} "-output" {{.*}} "-output_no_suffix" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-GEN-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS2 %s
// CHK-TOOLS-CPU-OPTS2: opencl-aot{{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-CPU-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts=-DFOO1 -DFOO2"

// Sane-check "-compile-opts" and "-link-opts" passing for multiple targets
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown,spir64_gen-unknown-unknown \
// RUN:   -Xsycl-target-backend=spir64_gen-unknown-unknown "-device skl -cl-opt-disable" -Xsycl-target-linker=spir64-unknown-unknown "-cl-denorms-are-zero" %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-TOOLS-MULT-OPTS,CHK-TOOLS-MULT-OPTS-NEG %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device skl -cl-opt-disable" -Xsycl-target-linker=spir64 "-cl-denorms-are-zero" %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-TOOLS-MULT-OPTS,CHK-TOOLS-MULT-OPTS-NEG %s
// CHK-TOOLS-MULT-OPTS: clang-offload-wrapper{{.*}} "-link-opts=-cl-denorms-are-zero"{{.*}} "-target=spir64"
// CHK-TOOLS-MULT-OPTS: ocloc{{.*}} "-device" "skl"{{.*}} "-cl-opt-disable"
// CHK-TOOLS-MULT-OPTS-NEG-NOT: clang-offload-wrapper{{.*}} "-compile-opts=-device skl -cl-opt-disable"{{.*}} "-target=spir64"
// CHK-TOOLS-MULT-OPTS-NEG-NOT: clang-offload-wrapper{{.*}} "-link-opts=-cl-denorms-are-zero"{{.*}} "-target=spir64_gen"

/// ###########################################################################

/// offload with multiple targets, including AOT
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown,spir64_fpga-unknown-unknown,spir64_gen-unknown-unknown -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG: 9: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 11: compiler, {10}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 12: linker, {11}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: offload, "device-sycl (spir64-unknown-unknown)" {17}, object
// CHK-PHASE-MULTI-TARG: 19: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: preprocessor, {19}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: compiler, {20}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: linker, {21}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: backend-compiler, {25}, fpga_aocx, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: file-table-tform, {23, 26}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: offload, "device-sycl (spir64_fpga-unknown-unknown)" {28}, object
// CHK-PHASE-MULTI-TARG: 30: linker, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: sycl-post-link, {30}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: backend-compiler, {33}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG: 35: file-table-tform, {31, 34}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 37: offload, "device-sycl (spir64_gen-unknown-unknown)" {36}, object
// CHK-PHASE-MULTI-TARG: 38: linker, {8, 18, 29, 37}, image, (host-sycl)

/// Options should not be duplicated in AOT calls
// RUN: %clang -fsycl -### -fsycl-targets=spir64_fpga -Xshardware -Xsycl-target-backend "-DBLAH" %s 2>&1 \
// RUN:  | FileCheck -check-prefix=DUP-OPT %s
// DUP-OPT-NOT: aoc{{.*}} "-DBLAH" {{.*}} "-DBLAH"
