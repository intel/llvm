///
/// Device code split specific test.
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-use-bitcode
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fno-sycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fno-sycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split=per_source -fsycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-CL-MODE: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES: 6: append-footer, {5}, c++, (host-sycl)
// CHK-PHASES: 7: preprocessor, {6}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASES: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASES: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASES: 11: linker, {10}, image, (host-sycl)
// CHK-PHASES: 12: linker, {4}, ir, (device-sycl)
// CHK-PHASES: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASES: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASES: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASES: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASES: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {17}, image
// CHK-PHASES-CL-MODE: 18: offload, "host-sycl (x86_64-pc-windows-msvc)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {17}, image

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-LIB: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LIB: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LIB: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-LIB: 7: append-footer, {6}, c++, (host-sycl)
// CHK-PHASES-LIB: 8: preprocessor, {7}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 9: compiler, {8}, ir, (host-sycl)
// CHK-PHASES-LIB: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES-LIB: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES-LIB: 12: linker, {0, 11}, image, (host-sycl)
// CHK-PHASES-LIB: 13: linker, {5}, ir, (device-sycl)
// CHK-PHASES-LIB: 14: sycl-post-link, {13}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 15: file-table-tform, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 17: file-table-tform, {14, 16}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-PHASES-LIB: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64-unknown-unknown-sycldevice)" {18}, image

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-use-footer -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s

// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: input, "[[INPUT1]]", c++, (device-sycl)
// CHK-PHASES-FILES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FILES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-FILES: 7: append-footer, {6}, c++, (host-sycl)
// CHK-PHASES-FILES: 8: preprocessor, {7}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 9: compiler, {8}, ir, (host-sycl)
// CHK-PHASES-FILES: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES-FILES: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES-FILES: 12: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 14: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-PHASES-FILES: 15: preprocessor, {14}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (device-sycl)
// CHK-PHASES-FILES: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, c++-cpp-output
// CHK-PHASES-FILES: 18: append-footer, {17}, c++, (host-sycl)
// CHK-PHASES-FILES: 19: preprocessor, {18}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 20: compiler, {19}, ir, (host-sycl)
// CHK-PHASES-FILES: 21: backend, {20}, assembler, (host-sycl)
// CHK-PHASES-FILES: 22: assembler, {21}, object, (host-sycl)
// CHK-PHASES-FILES: 23: linker, {0, 11, 22}, image, (host-sycl)
// CHK-PHASES-FILES: 24: linker, {5, 16}, ir, (device-sycl)
// CHK-PHASES-FILES: 25: sycl-post-link, {24}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 26: file-table-tform, {25}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 27: llvm-spirv, {26}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 28: file-table-tform, {25, 27}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 29: clang-offload-wrapper, {28}, object, (device-sycl)
// CHK-PHASES-FILES: 30: offload, "host-sycl (x86_64-unknown-linux-gnu)" {23}, "device-sycl (spir64-unknown-unknown-sycldevice)" {29}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t_dir/dummy 2>&1 \
// RUN:   | FileCheck -DINPUT=%t_dir/dummy -check-prefix=CHK-UBACTIONS %s
// CHK-UBACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBACTIONS: 1: input, "[[INPUT]]", object, (host-sycl)
// CHK-UBACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBACTIONS: 3: linker, {0, 2}, image, (host-sycl)
// CHK-UBACTIONS: 4: linker, {2}, ir, (device-sycl)
// CHK-UBACTIONS: 5: sycl-post-link, {4}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 6: file-table-tform, {5}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 7: llvm-spirv, {6}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 8: file-table-tform, {5, 7}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 9: clang-offload-wrapper, {8}, object, (device-sycl)
// CHK-UBACTIONS: 10: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {9}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling with source
// RUN:   touch %t.o
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-use-footer -fno-sycl-device-lib=all -fsycl-device-code-split %t.o -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-UBUACTIONS: 4: preprocessor, {3}, c++-cpp-output, (host-sycl)
// CHK-UBUACTIONS: 5: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-UBUACTIONS: 6: preprocessor, {5}, c++-cpp-output, (device-sycl)
// CHK-UBUACTIONS: 7: compiler, {6}, ir, (device-sycl)
// CHK-UBUACTIONS: 8: offload, "host-sycl (x86_64-unknown-linux-gnu)" {4}, "device-sycl (spir64-unknown-unknown-sycldevice)" {7}, c++-cpp-output
// CHK-UBUACTIONS: 9: append-footer, {8}, c++, (host-sycl)
// CHK-UBUACTIONS: 10: preprocessor, {9}, c++-cpp-output, (host-sycl)
// CHK-UBUACTIONS: 11: compiler, {10}, ir, (host-sycl)
// CHK-UBUACTIONS: 12: backend, {11}, assembler, (host-sycl)
// CHK-UBUACTIONS: 13: assembler, {12}, object, (host-sycl)
// CHK-UBUACTIONS: 14: linker, {0, 2, 13}, image, (host-sycl)
// CHK-UBUACTIONS: 15: linker, {2, 7}, ir, (device-sycl)
// CHK-UBUACTIONS: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 19: file-table-tform, {16, 18}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-UBUACTIONS: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {14}, "device-sycl (spir64-unknown-unknown-sycldevice)" {20}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-AOT: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-AOT: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-AOT: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES-AOT: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-FPGA: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-GEN: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-CPU: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-AOT: 6: append-footer, {5}, c++, (host-sycl)
// CHK-PHASES-AOT: 7: preprocessor, {6}, c++-cpp-output, (host-sycl)
// CHK-PHASES-AOT: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASES-AOT: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASES-AOT: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASES-AOT: 11: linker, {10}, image, (host-sycl)
// CHK-PHASES-AOT: 12: linker, {4}, ir, (device-sycl)
// CHK-PHASES-AOT: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 16: backend-compiler, {15}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 17: file-table-tform, {13, 16}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-PHASES-AOT-FPGA: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {18}, image
// CHK-PHASES-AOT-GEN: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {18}, image
// CHK-PHASES-AOT-CPU: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {18}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-AOT: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "-split=auto" {{.*}} "-spec-const=default" "-o" "[[OUTPUT3:.+\.table]]" "[[OUTPUT2]]"
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
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" {{.*}} "-o" "[[TMPII:.+\.ii]]"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-o" "[[OUTPUT10:.+\.o]]"
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT10]]" "[[OUTPUT9]]" {{.*}} "-lsycl"

/// ###########################################################################

/// offload with multiple targets, including AOT
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl-use-footer -fsycl -fno-sycl-device-lib=all -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64_fpga-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)

// CHK-PHASE-MULTI-TARG: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG: 6: append-footer, {5}, c++, (host-sycl)
// CHK-PHASE-MULTI-TARG: 7: preprocessor, {6}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG: 11: linker, {10}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG: 12: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 13: preprocessor, {12}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 14: compiler, {13}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: linker, {14}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 19: file-table-tform, {16, 18}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: preprocessor, {21}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: compiler, {22}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: linker, {23}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: sycl-post-link, {24}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: file-table-tform, {25}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: llvm-spirv, {26}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: backend-compiler, {27}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: file-table-tform, {25, 28}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 30: clang-offload-wrapper, {29}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: linker, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: sycl-post-link, {31}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: file-table-tform, {32}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: llvm-spirv, {33}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 35: backend-compiler, {34}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 36: file-table-tform, {32, 35}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 37: clang-offload-wrapper, {36}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 38: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {20}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {30}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {37}, image

// Check -fsycl-device-code-split=per_kernel option passing.
// RUN:   %clang -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// CHK-ONE-KERNEL: sycl-post-link{{.*}} "-split=kernel"{{.*}} "-o"{{.*}}

// Check -fsycl-device-code-split=per_source option passing.
// RUN:   %clang -### -fsycl -fsycl-device-code-split=per_source %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split=per_source %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PER-SOURCE
// CHK-PER-SOURCE: sycl-post-link{{.*}} "-split=source"{{.*}} "-o"{{.*}}

// Check -fsycl-device-code-split option passing.
// RUN:   %clang -### -fsycl -fsycl-device-code-split %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl -fsycl-device-code-split=auto %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split=auto %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// RUN:   %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-AUTO
// CHK-AUTO: sycl-post-link{{.*}} "-split=auto"{{.*}} "-o"{{.*}}

// Check no device code split mode.
// RUN:   %clang -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// CHK-NO-SPLIT-NOT: sycl-post-link{{.*}} -split{{.*}}

// Check ESIMD device code split.
// RUN:   %clang    -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang    -### -fsycl -fsycl-device-code-split-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang    -### -fsycl -fno-sycl-device-code-split-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-ESIMD-SPLIT
// RUN:   %clang_cl -### -fsycl -fno-sycl-device-code-split-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-ESIMD-SPLIT
// RUN:   %clang    -### -fsycl -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang    -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// RUN:   %clang_cl -### -fsycl -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-SPLIT
// CHK-ESIMD-SPLIT: sycl-post-link{{.*}} "-split-esimd"
// CHK-NO-ESIMD-SPLIT-NOT: sycl-post-link{{.*}} "-split-esimd"

// Check lowering of ESIMD device code.
// RUN:   %clang    -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang    -### -fsycl -fsycl-device-code-lower-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-lower-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang    -### -fsycl -fno-sycl-device-code-lower-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-ESIMD-LOWER
// RUN:   %clang_cl -### -fsycl -fno-sycl-device-code-lower-esimd %s 2>&1 | FileCheck %s -check-prefixes=CHK-NO-ESIMD-LOWER
// RUN:   %clang    -### -fsycl -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang    -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// RUN:   %clang_cl -### -fsycl -fintelfpga %s 2>&1 | FileCheck %s -check-prefixes=CHK-ESIMD-LOWER
// CHK-ESIMD-LOWER: sycl-post-link{{.*}} "-lower-esimd"
// CHK-NO-ESIMD-LOWER-NOT: sycl-post-link{{.*}} "-lower-esimd"
