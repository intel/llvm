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
// RUN:   %clang_cl -ccc-print-phases -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64-unknown-unknown-sycldevice-coff %s 2>&1 \
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
// CHK-PHASES-CL-MODE: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice-coff)" {4}, cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES: 11: linker, {10}, ir, (device-sycl)
// CHK-PHASES: 12: sycl-post-link, {11}, tempentriesfilelist, (device-sycl)
// CHK-PHASES: 13: sycl-post-link, {11}, tempfilelist, (device-sycl)
// CHK-PHASES: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES: 15: clang-offload-wrapper, {12, 14}, object, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {15}, image
// CHK-PHASES-CL-MODE: 16: offload, "host-sycl (x86_64-pc-windows-msvc)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice-coff)" {15}, image

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
// CHK-PHASES-LIB: 13: sycl-post-link, {12}, tempentriesfilelist, (device-sycl)
// CHK-PHASES-LIBL 14: sycl-post-link, {12}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 16: clang-offload-wrapper, {13, 15}, object, (device-sycl)
// CHK-PHASES-LIB: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image

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
// CHK-PHASES-FILES: 23: sycl-post-link, {22}, tempentriesfilelist, (device-sycl)
// CHK-PHASES-FILES: 24: sycl-post-link, {22}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 26: clang-offload-wrapper, {23, 25}, object, (device-sycl)
// CHK-PHASES-FILES: 27: offload, "host-sycl (x86_64-unknown-linux-gnu)" {19}, "device-sycl (spir64-unknown-unknown-sycldevice)" {26}, image

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
// CHK-UBACTIONS: 5: sycl-post-link, {4}, tempentriesfilelist, (device-sycl)
// CHK-UBACTIONS: 6: sycl-post-link, {4}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 7: llvm-spirv, {6}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 8: clang-offload-wrapper, {5, 7}, object, (device-sycl)
// CHK-UBACTIONS: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {8}, image

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
// CHK-UBUACTIONS: 15: sycl-post-link, {14}, tempentriesfilelist, (device-sycl)
// CHK-UBUACTIONS: 16: sycl-post-link, {14}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 18: clang-offload-wrapper, {15, 17}, object, (device-sycl)
// CHK-UBUACTIONS: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64-unknown-unknown-sycldevice)" {18}, image

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
// CHK-PHASES-AOT: 12: sycl-post-link, {11}, tempentriesfilelist, (device-sycl)
// CHK-PHASES-AOT: 13: sycl-post-link, {11}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-GEN: 15: backend-compiler, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-FPGA: 15: backend-compiler, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 16: clang-offload-wrapper, {12, 15}, object, (device-sycl)
// CHK-PHASES-FPGA: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {16}, image
// CHK-PHASES-GEN: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {16}, image
// CHK-PHASES-CPU: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {16}, image

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
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "[[OUTPUT2]]" "-txt-files-list=[[OUTPUT6:.+\.txt]]" "-o"
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "[[OUTPUT2]]" "-ir-files-list=[[OUTPUT3:.+\.txt]]" "-o"
// CHK-TOOLS-AOT: llvm-foreach{{.*}}  "--in-file-list=[[OUTPUT3]]" "--in-replace=[[OUTPUT3]]" "--out-ext=spv" "--out-file-list=[[OUTPUT4:.+\.txt]]" "--out-replace=[[OUTPUT4]]" "--" "{{.*}}llvm-spirv{{.*}} "-o" "[[OUTPUT4]]" {{.*}} "[[OUTPUT3]]"
// CHK-TOOLS-FPGA: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT5:.+\.txt]]{{.*}} "--" "{{.*}}aoc{{.*}} "-o" "[[OUTPUT5]]" "[[OUTPUT4]]"
// CHK-TOOLS-GEN: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT5:.+\.txt]]{{.*}} "--" "{{.*}}ocloc{{.*}} "-output" "[[OUTPUT5]]" "-file" "[[OUTPUT4]]"
// CHK-TOOLS-CPU: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT5:.+\.txt]]{{.*}} "--" "{{.*}}opencl-aot{{.*}} "-o=[[OUTPUT5]]" "--device=cpu" "[[OUTPUT4]]"
// CHK-TOOLS-FPGA: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT7:.+\.txt]]{{.*}} "--" "{{.*}}clang-offload-wrapper{{.*}} "-o={{.*}}" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga{{.*}}" "-kind=sycl" "-entries=[[OUTPUT6]]" "[[OUTPUT5]]"
// CHK-TOOLS-GEN: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT7:.+\.txt]]{{.*}} "--" "{{.*}}clang-offload-wrapper{{.*}} "-o={{.*}}" "-host=x86_64-unknown-linux-gnu" "-target=spir64_gen{{.*}}" "-kind=sycl" "-entries=[[OUTPUT6]]" "[[OUTPUT5]]"
// CHK-TOOLS-CPU: llvm-foreach{{.*}} "--out-file-list=[[OUTPUT7:.+\.txt]]{{.*}} "--" "{{.*}}clang-offload-wrapper{{.*}} "-o={{.*}}" "-host=x86_64-unknown-linux-gnu" "-target=spir64_x86_64{{.*}}" "-kind=sycl" "-entries=[[OUTPUT6]]" "[[OUTPUT5]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "-o" "[[OUTPUT8:.+\.bc]]" "@[[OUTPUT7]]"
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
// CHK-PHASE-MULTI-TARG: 14: sycl-post-link, {13}, tempentriesfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: sycl-post-link, {13}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: clang-offload-wrapper, {14, 16}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 19: preprocessor, {18}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: compiler, {19}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: linker, {20}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: sycl-post-link, {21}, tempentriesfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: sycl-post-link, {21}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: llvm-spirv, {23}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: backend-compiler, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: clang-offload-wrapper, {22, 25}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: linker, {27}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: sycl-post-link, {28}, tempentriesfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 30: sycl-post-link, {28}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: llvm-spirv, {30}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: backend-compiler, {31}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: clang-offload-wrapper, {29, 32}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-unknown-sycldevice)" {17}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {26}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {33}, image


// Check -fsycl-one-kernel-per-module option passing.
// RUN:   %clang -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split=per_kernel %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-ONE-KERNEL
// CHK-ONE-KERNEL: sycl-post-link{{.*}} "-txt-files-list{{.*}} "-one-kernel"
// CHK-ONE-KERNEL: sycl-post-link{{.*}} "-ir-files-list{{.*}} "-one-kernel"

// Check no device code split mode.
// RUN:   %clang -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// RUN:   %clang_cl -### -fsycl -fsycl-device-code-split -fsycl-device-code-split=off %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-NO-SPLIT
// CHK-NO-SPLIT-NOT: sycl-post-link{{.*}} "-txt-files-list{{.*}}
// CHK-NO-SPLIT-NOT: sycl-post-link{{.*}} "-ir-files-list{{.*}}
// CHK-NO-SPLIT-NOT: llvm-foreach{{.*}}

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
