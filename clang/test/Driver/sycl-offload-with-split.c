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
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split=per_source -fno-sycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases -fsycl -fsycl-device-code-split=per_source -fno-sycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split=per_source -fsycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases -fsycl -fsycl-device-code-split=per_source -fsycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASES-CL-MODE: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES: 11: linker, {10}, ir, (device-sycl)
// CHK-PHASES: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-PHASES: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-PHASES: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES: 15: file-table-tform, {12, 14}, tempfiletable, (device-sycl)
// CHK-PHASES: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image
// CHK-PHASES-CL-MODE: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES-LIB: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-LIB: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-LIB: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, cpp-output
// CHK-PHASES-LIB: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-LIB: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-LIB: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-LIB: 10: linker, {0, 9}, image, (host-sycl)
// CHK-PHASES-LIB: 11: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LIB: 12: linker, {11}, ir, (device-sycl)
// CHK-PHASES-LIB: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-LIB: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {17}, image

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s

// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: input, "[[INPUT1]]", c, (device-sycl)
// CHK-PHASES-FILES: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, cpp-output
// CHK-PHASES-FILES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-FILES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-FILES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-FILES: 10: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 11: preprocessor, {10}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 12: input, "[[INPUT2]]", c, (device-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 14: compiler, {13}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {14}, cpp-output
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (host-sycl)
// CHK-PHASES-FILES: 17: backend, {16}, assembler, (host-sycl)
// CHK-PHASES-FILES: 18: assembler, {17}, object, (host-sycl)
// CHK-PHASES-FILES: 19: linker, {0, 9, 18}, image, (host-sycl)
// CHK-PHASES-FILES: 20: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FILES: 21: compiler, {13}, ir, (device-sycl)
// CHK-PHASES-FILES: 22: linker, {20, 21}, ir, (device-sycl)
// CHK-PHASES-FILES: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 26: file-table-tform, {23, 25}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHK-PHASES-FILES: 28: offload, "host-sycl (x86_64-unknown-linux-gnu)" {19}, "device-sycl (spir64-unknown-unknown-sycldevice)" {27}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t_dir/dummy 2>&1 \
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
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-device-code-split %t.o -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-UBUACTIONS: 4: preprocessor, {3}, cpp-output, (host-sycl)
// CHK-UBUACTIONS: 5: input, "[[INPUT2]]", c, (device-sycl)
// CHK-UBUACTIONS: 6: preprocessor, {5}, cpp-output, (device-sycl)
// CHK-UBUACTIONS: 7: compiler, {6}, sycl-header, (device-sycl)
// CHK-UBUACTIONS: 8: offload, "host-sycl (x86_64-unknown-linux-gnu)" {4}, "device-sycl (spir64-unknown-unknown-sycldevice)" {7}, cpp-output
// CHK-UBUACTIONS: 9: compiler, {8}, ir, (host-sycl)
// CHK-UBUACTIONS: 10: backend, {9}, assembler, (host-sycl)
// CHK-UBUACTIONS: 11: assembler, {10}, object, (host-sycl)
// CHK-UBUACTIONS: 12: linker, {0, 2, 11}, image, (host-sycl)
// CHK-UBUACTIONS: 13: compiler, {6}, ir, (device-sycl)
// CHK-UBUACTIONS: 14: linker, {2, 13}, ir, (device-sycl)
// CHK-UBUACTIONS: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-UBUACTIONS: 20: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64-unknown-unknown-sycldevice)" {19}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES-AOT: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASES-AOT: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES-AOT: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASES-AOT: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES-FPGA: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASES-GEN: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASES-CPU: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASES-AOT: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-AOT: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-AOT: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-AOT: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-AOT: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-AOT: 11: linker, {10}, ir, (device-sycl)
// CHK-PHASES-AOT: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-GEN: 15: backend-compiler, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-FPGA: 15: backend-compiler, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-CPU: 15: backend-compiler, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 16: file-table-tform, {12, 15}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-FPGA: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {17}, image
// CHK-PHASES-GEN: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {17}, image
// CHK-PHASES-CPU: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {17}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-AOT: clang{{.*}} "-fsycl-is-device" {{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "-spec-const=default" "-o" "[[OUTPUT3:.+\.table]]" "[[OUTPUT2]]"
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
// CHK-TOOLS-FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-GEN: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT10:.+\.o]]"
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT10]]" "[[OUTPUT9]]" {{.*}} "-lsycl"

/// ###########################################################################

/// offload with multiple targets, including AOT
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-code-split -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64_fpga-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice -###  -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASE-MULTI-TARG: 1: preprocessor, {0}, cpp-output, (host-sycl)

// CHK-PHASE-MULTI-TARG: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASE-MULTI-TARG: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, cpp-output
// CHK-PHASE-MULTI-TARG: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG: 9: linker, {8}, image, (host-sycl)

// CHK-PHASE-MULTI-TARG: 10: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 11: preprocessor, {10}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 12: compiler, {11}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 13: linker, {12}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 14: sycl-post-link, {13}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: file-table-tform, {14}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: file-table-tform, {14, 16}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: clang-offload-wrapper, {17}, object, (device-sycl)

// CHK-PHASE-MULTI-TARG: 19: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: preprocessor, {19}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: compiler, {20}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: linker, {21}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: backend-compiler, {25}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: file-table-tform, {23, 26}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: clang-offload-wrapper, {27}, object, (device-sycl)

// CHK-PHASE-MULTI-TARG: 29: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 30: linker, {29}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: sycl-post-link, {30}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: backend-compiler, {33}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 35: file-table-tform, {31, 34}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 37: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {18}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {28}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {36}, image

// Check -fsycl-one-kernel-per-module option passing.
// RUN:   %clang -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// CHK-ONE-KERNEL: sycl-post-link{{.*}} "-split=kernel"{{.*}} "-o"{{.*}}

// Check no device code split mode.
// RUN:   %clang -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// CHK-NO-SPLIT-NOT: sycl-post-link{{.*}} -split{{.*}}

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
