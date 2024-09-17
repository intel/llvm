///
/// Device code split specific test.
///

// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-device-obj=llvmir
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-targets=spir64-unknown-unknown -- %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-device-obj=spirv %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-device-obj=spirv -- %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-device-obj=llvmir %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-device-obj=llvmir -- %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-CL-MODE: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {4}, ir, (device-sycl)
// CHK-PHASES: 10: sycl-post-link, {9}, tempfiletable, (device-sycl)
// CHK-PHASES: 11: file-table-tform, {10}, tempfilelist, (device-sycl)
// CHK-PHASES: 12: llvm-spirv, {11}, tempfilelist, (device-sycl)
// CHK-PHASES: 13: file-table-tform, {10, 12}, tempfiletable, (device-sycl)
// CHK-PHASES: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-PHASES: 15: offload, "device-sycl (spir64-unknown-unknown)" {14}, object
// CHK-PHASES: 16: linker, {8, 15}, image, (host-sycl)

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-LIB: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LIB: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LIB: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHK-PHASES-LIB: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-LIB: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-LIB: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-LIB: 10: linker, {5}, ir, (device-sycl)
// CHK-PHASES-LIB: 11: sycl-post-link, {10}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 12: file-table-tform, {11}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 13: llvm-spirv, {12}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 14: file-table-tform, {11, 13}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES-LIB: 16: offload, "device-sycl (spir64-unknown-unknown)" {15}, object
// CHK-PHASES-LIB: 17: linker, {0, 9, 16}, image, (host-sycl)

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s

// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: input, "[[INPUT1]]", c++, (device-sycl)
// CHK-PHASES-FILES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FILES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHK-PHASES-FILES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-FILES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-FILES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-FILES: 10: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 11: preprocessor, {10}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 12: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 14: compiler, {13}, ir, (device-sycl)
// CHK-PHASES-FILES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown)" {14}, c++-cpp-output
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (host-sycl)
// CHK-PHASES-FILES: 17: backend, {16}, assembler, (host-sycl)
// CHK-PHASES-FILES: 18: assembler, {17}, object, (host-sycl)
// CHK-PHASES-FILES: 19: linker, {5, 14}, ir, (device-sycl)
// CHK-PHASES-FILES: 20: sycl-post-link, {19}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 21: file-table-tform, {20}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 22: llvm-spirv, {21}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 23: file-table-tform, {20, 22}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 24: clang-offload-wrapper, {23}, object, (device-sycl)
// CHK-PHASES-FILES: 25: offload, "device-sycl (spir64-unknown-unknown)" {24}, object
// CHK-PHASES-FILES: 26: linker, {0, 9, 18, 25}, image, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t_dir/dummy 2>&1 \
// RUN:   | FileCheck -DINPUT=%t_dir/dummy -check-prefix=CHK-UBACTIONS %s
// CHK-UBACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBACTIONS: 1: input, "[[INPUT]]", object, (host-sycl)
// CHK-UBACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBACTIONS: 3: spirv-to-ir-wrapper, {2}, ir, (device-sycl)
// CHK-UBACTIONS: 4: linker, {3}, ir, (device-sycl)
// CHK-UBACTIONS: 5: sycl-post-link, {4}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 6: file-table-tform, {5}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 7: llvm-spirv, {6}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 8: file-table-tform, {5, 7}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 9: clang-offload-wrapper, {8}, object, (device-sycl)
// CHK-UBACTIONS: 10: offload, "device-sycl (spir64-unknown-unknown)" {9}, object
// CHK-UBACTIONS: 11: linker, {0, 2, 10}, image, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling with source
// RUN:   touch %t.o
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split %t.o -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-UBUACTIONS: 4: preprocessor, {3}, c++-cpp-output, (host-sycl)
// CHK-UBUACTIONS: 5: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-UBUACTIONS: 6: preprocessor, {5}, c++-cpp-output, (device-sycl)
// CHK-UBUACTIONS: 7: compiler, {6}, ir, (device-sycl)
// CHK-UBUACTIONS: 8: offload, "host-sycl (x86_64-unknown-linux-gnu)" {4}, "device-sycl (spir64-unknown-unknown)" {7}, c++-cpp-output
// CHK-UBUACTIONS: 9: compiler, {8}, ir, (host-sycl)
// CHK-UBUACTIONS: 10: backend, {9}, assembler, (host-sycl)
// CHK-UBUACTIONS: 11: assembler, {10}, object, (host-sycl)
// CHK-UBUACTIONS: 12: spirv-to-ir-wrapper, {2}, ir, (device-sycl)
// CHK-UBUACTIONS: 13: linker, {12, 7}, ir, (device-sycl)
// CHK-UBUACTIONS: 14: sycl-post-link, {13}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 15: file-table-tform, {14}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 17: file-table-tform, {14, 16}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-UBUACTIONS: 19: offload, "device-sycl (spir64-unknown-unknown)" {18}, object
// CHK-UBUACTIONS: 20: linker, {0, 2, 11, 19}, image, (host-sycl)

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
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
// CHK-PHASES-AOT: 13: backend-compiler, {12}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 14: file-table-tform, {10, 13}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES-FPGA: 16: offload, "device-sycl (spir64_fpga-unknown-unknown)" {15}, object
// CHK-PHASES-GEN: 16: offload, "device-sycl (spir64_gen-unknown-unknown)" {15}, object
// CHK-PHASES-CPU: 16: offload, "device-sycl (spir64_x86_64-unknown-unknown)" {15}, object
// CHK-PHASES-AOT: 17: linker, {8, 16}, image, (host-sycl)

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-AOT: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-o" "[[OUTPUT10:.+\.o]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-FPGA: sycl-post-link{{.*}} "-split=auto"{{.*}} "-spec-const=emulation"{{.*}} "-o" "[[OUTPUT3:.+\.table]]" "[[OUTPUT2]]"
// CHK-TOOLS-GEN: sycl-post-link{{.*}} "-split=auto"{{.*}} "-spec-const=emulation"{{.*}} "-o" "[[OUTPUT3:.+\.table]]" "[[OUTPUT2]]"
// CHK-TOOLS-CPU: sycl-post-link{{.*}} "-split=auto"{{.*}} "-spec-const=emulation"{{.*}} "-o" "spir64_x86_64,[[OUTPUT3:.+\.table]]" "[[OUTPUT2]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-o" "[[OUTPUT4:.+\.txt]]" "[[OUTPUT3]]"
// CHK-TOOLS-AOT: llvm-foreach{{.*}} "--in-file-list=[[OUTPUT4]]" "--in-replace=[[OUTPUT4]]" "--out-ext=spv" "--out-file-list=[[OUTPUT5:.+\.txt]]" "--out-replace=[[OUTPUT5]]" "--" "{{.*}}llvm-spirv{{.*}}" "-o" "[[OUTPUT5]]" {{.*}} "[[OUTPUT4]]"
// CHK-TOOLS-FPGA: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT6:.+\.txt]]{{.*}} "--" "{{.*}}aoc{{.*}} "-o" "[[OUTPUT6]]" "[[OUTPUT5]]"
// CHK-TOOLS-GEN: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT6:.+\.txt]]{{.*}} "--" "{{.*}}ocloc{{.*}} "-output" "[[OUTPUT6]]" "-file" "[[OUTPUT5]]"
// CHK-TOOLS-CPU: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT6:.+\.txt]]{{.*}} "--" "{{.*}}opencl-aot{{.*}} "-o=[[OUTPUT6]]" "--device=cpu" "[[OUTPUT5]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-o" "[[OUTPUT7:.+\.table]]" "[[OUTPUT3]]" "[[OUTPUT6]]"
// CHK-TOOLS-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT8:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga" "-kind=sycl" "-batch" "[[OUTPUT7]]"
// CHK-TOOLS-GEN: clang-offload-wrapper{{.*}} "-o=[[OUTPUT8:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_gen" "-kind=sycl" "-batch" "[[OUTPUT7]]"
// CHK-TOOLS-CPU: clang-offload-wrapper{{.*}} "-o=[[OUTPUT8:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_x86_64" "-kind=sycl" "-batch" "[[OUTPUT7]]"
// CHK-TOOLS-AOT: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT9:.+\.o]]" "[[OUTPUT8]]"
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT10]]" "[[OUTPUT9]]" {{.*}} "-lsycl"

/// ###########################################################################

/// Check parallel compilation enforcement for split modules when running SPIR-V translation and AOT compilation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-max-parallel-link-jobs=4 -fsycl-targets=spir64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-PARALLEL-JOBS
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-max-parallel-link-jobs=4 -fsycl-targets=spir64_fpga-unknown-unknown -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-PARALLEL-JOBS,CHK-PARALLEL-JOBS-AOT -DBE_COMPILER=aoc
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl-max-parallel-link-jobs=4 -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-PARALLEL-JOBS,CHK-PARALLEL-JOBS-AOT -DBE_COMPILER=aoc
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-max-parallel-link-jobs=4 -fsycl-targets=spir64_gen-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-PARALLEL-JOBS,CHK-PARALLEL-JOBS-AOT -DBE_COMPILER=ocloc
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-max-parallel-link-jobs=4 -fsycl-targets=spir64_x86_64-unknown-unknown %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-PARALLEL-JOBS,CHK-PARALLEL-JOBS-AOT -DBE_COMPILER=opencl-aot
// CHK-PARALLEL-JOBS: llvm-foreach{{.*}} "--jobs=4" "--" "{{.*}}llvm-spirv{{.*}}"
// CHK-PARALLEL-JOBS-AOT: llvm-foreach{{.*}} "--jobs=4" "--" "{{.*}}[[BE_COMPILER]]{{.*}}

/// ###########################################################################

/// offload with multiple targets, including AOT
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown,spir64_fpga-unknown-unknown,spir64_gen-unknown-unknown -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
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
// CHK-PHASE-MULTI-TARG: 26: backend-compiler, {25}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: file-table-tform, {23, 26}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: offload, "device-sycl (spir64_fpga-unknown-unknown)" {28}, object
// CHK-PHASE-MULTI-TARG: 30: linker, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: sycl-post-link, {30}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: backend-compiler, {33}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 35: file-table-tform, {31, 34}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 37: offload, "device-sycl (spir64_gen-unknown-unknown)" {36}, object
// CHK-PHASE-MULTI-TARG: 38: linker, {8, 18, 29, 37}, image, (host-sycl)

// Check -fsycl-device-code-split=per_kernel option passing.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_kernel -- %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// CHK-ONE-KERNEL: sycl-post-link{{.*}} "-split=kernel"{{.*}} "-o"{{.*}}

// Check -fsycl-device-code-split=per_source option passing.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_source %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_source -- %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// CHK-PER-SOURCE: sycl-post-link{{.*}} "-split=source"{{.*}} "-o"{{.*}}

// Check -fsycl-device-code-split option passing.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-code-split -- %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=auto %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-code-split=auto -- %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -- %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// CHK-AUTO: sycl-post-link{{.*}} "-split=auto"{{.*}} "-o"{{.*}}

// Check no device code split mode.
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-code-split -fsycl-device-code-split=off -- %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// CHK-NO-SPLIT-NOT: sycl-post-link{{.*}} "-split={{.*}}

// Check no device code split mode is passed to sycl-post-link when -fsycl-device-code-split is not set and the target is FPGA
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-SPLIT

// Check device code split mode is honored when -fsycl-device-code-split is set and the target is FPGA
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=auto -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=per_source -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split=off -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-SPLIT

// Check ESIMD device code split.
// RUN:   %clang    -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -- %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang    -### -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang    -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang_cl -### -fintelfpga -- %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// CHK-ESIMD-SPLIT: sycl-post-link{{.*}} "-split-esimd"

// Check lowering of ESIMD device code.
// RUN:   %clang    -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -- %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang    -### -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang    -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_fpga-unknown-unknown %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang_cl -### -fintelfpga -- %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// CHK-ESIMD-LOWER: sycl-post-link{{.*}} "-lower-esimd"

// Check -f[no]sycl-device-code-split-esimd option's effect on sycl-post-link invocation
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-code-split-esimd %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT-ON
// RUN:   %clang -### -fsycl --no-offload-new-driver -fno-sycl-device-code-split-esimd %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT-OFF
// RUN:   %clang -### -fsycl --no-offload-new-driver %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT-DEFAULT
// RUN:   %clang -### -fsycl --no-offload-new-driver -nocudalib -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT-NON-SPIRV
// CHK-ESIMD-SPLIT-ON: sycl-post-link{{.*}} "-split-esimd"{{.*}} "-o"{{.*}}
// CHK-ESIMD-SPLIT-OFF-NOT: sycl-post-link{{.*}} "-split-esimd"{{.*}}
// CHK-ESIMD-SPLIT-DEFAULT: sycl-post-link{{.*}} "-split-esimd"{{.*}} "-o"{{.*}}
// CHK-ESIMD-SPLIT-NON-SPIRV-NOT: sycl-post-link{{.*}} "-split-esimd"{{.*}}
