///
/// Perform several driver tests for SYCL offloading
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   %clang -### -fsycl -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: SYCL target is invalid: 'aaa-bbb-ccc-ddd'
// RUN:   %clang -### -fsycl -fsycl-add-targets=dummy-target:dummy-file %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-ADD %s
// RUN:   %clang_cl -### -fsycl -fsycl-add-targets=dummy-target:dummy-file %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET-ADD %s
// CHK-INVALID-TARGET-ADD: error: SYCL target is invalid: 'dummy-target'

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   %clang -### -fsycl -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// CHK-INVALID-REAL-TARGET: error: SYCL target is invalid: 'x86_64'

/// ###########################################################################

/// Check warning for empty -fsycl-targets
// RUN:   %clang -### -fsycl -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// CHK-EMPTY-SYCLTARGETS: warning: joined argument expects additional value: '-fsycl-targets='

/// ###########################################################################

/// Check error for no -fsycl option
// RUN:   %clang -### -fsycl-targets=spir64-unknown-unknown-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   %clang_cl -### -fsycl-targets=spir64-unknown-unknown-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-targets' must be used in conjunction with '-fsycl' to enable offloading
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-LINK-TGTS %s
// CHK-NO-FSYCL-LINK-TGTS: error: '-fsycl-link-targets' must be used in conjunction with '-fsycl' to enable offloading
// RUN:   %clang -### -fsycl-add-targets=spir64-unknown-unknown-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-ADD %s
// CHK-NO-FSYCL-ADD: error: '-fsycl-add-targets' must be used in conjunction with '-fsycl' to enable offloading
// RUN:   %clang -### -fsycl-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-LINK %s
// CHK-NO-FSYCL-LINK: error: '-fsycl-link' must be used in conjunction with '-fsycl' to enable offloading
// RUN:   %clang -### -fintelfpga  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-FINTELFPGA %s
// CHK-NO-FSYCL-FINTELFPGA: error: '-fintelfpga' must be used in conjunction with '-fsycl' to enable offloading

/// ###########################################################################

/// Check error for -fsycl-add-targets -fsycl-link-targets conflict
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// CHK-SYCL-ADD-LINK: error: The option -fsycl-link-targets= conflicts with -fsycl-add-targets=

/// ###########################################################################

/// Check error for -fsycl-targets -fsycl-link-targets conflict
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl-targets=spir64-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl-targets=spir64-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// CHK-SYCL-LINK-CONFLICT: error: The option -fsycl-targets= conflicts with -fsycl-link-targets=

/// ###########################################################################

/// Check error for -fsycl-targets -fintelfpga conflict
// RUN:   %clang -### -fsycl-targets=spir64-unknown-unknown-sycldevice -fintelfpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// RUN:   %clang_cl -### -fsycl-targets=spir64-unknown-unknown-sycldevice -fintelfpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// CHK-SYCL-FPGA-CONFLICT: error: The option -fsycl-targets= conflicts with -fintelfpga

/// ###########################################################################

/// Check error for -fsycl-targets with bad triple
// RUN:   %clang -### -fsycl-targets=spir64_bad-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-targets=spir64_bad-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-TRIPLE %s
// CHK-SYCL-FPGA-BAD-TRIPLE: error: SYCL target is invalid: 'spir64_bad-unknown-unknown-sycldevice'

/// Check no error for -fsycl-targets with good triple
// RUN:   %clang -### -fsycl-targets=spir-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64 -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_fpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_x86_64 -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64_gen -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang_cl -### -fsycl-targets=spir-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// CHK-SYCL-TARGET-NOT: error: SYCL target is invalid

/// Check error for -fsycl-[add|link]-targets with bad triple
// RUN:   %clang -### -fsycl-add-targets=spir64_bad-unknown-unknown-sycldevice:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-add-targets=spir64_bad-unknown-unknown-sycldevice:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64_bad-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64_bad-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE %s
// CHK-SYCL-FPGA-BAD-ADDLINK-TRIPLE: error: SYCL target is invalid: 'spir64_bad-unknown-unknown-sycldevice'

/// Check no error for -fsycl-[add|link]-targets with good triple
// RUN:   %clang -### -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy.spv -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-ADDLINK-TRIPLE %s
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-ADDLINK-TRIPLE %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-unknown-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-ADDLINK-TRIPLE %s
// CHK-SYCL-FPGA-ADDLINK-TRIPLE-NOT: error: SYCL target is invalid

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64-unknown-unknown-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: SYCL offloading target 'spir64-unknown-unknown-sycldevice' is similar to target 'spir64-unknown-unknown-sycldevice' already specified; will be ignored

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when multiple triples are used.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,spir-unknown-linux-sycldevice -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,spir-unknown-linux-sycldevice -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when an option requiring arguments is passed to it.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// CHK-FSYCL-COMPILER-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target-frontend argument: '-Xsycl-target-frontend -Xsycl-target-frontend', options requiring arguments are unsupported

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-use-bitcode
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-use-bitcode -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-use-bitcode -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-use-bitcode -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fsycl-use-bitcode -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-CL-MODE: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES: 11: linker, {5}, ir, (device-sycl)
// CHK-PHASES: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-PHASES: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-PHASES: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES: 15: file-table-tform, {12, 14}, tempfiletable, (device-sycl)
// CHK-PHASES: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image
// CHK-PHASES-CL-MODE: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image

/// ###########################################################################

/// Check the compilation flow to verify that the integrated header is filtered
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -c %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHK-INT-HEADER
// CHK-INT-HEADER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-INT-HEADER: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" "-dependency-filter" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT2:.+\.o]]"
// CHK-INT-HEADER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown-sycldevice,host-x86_64-unknown-linux-gnu" {{.*}} "-inputs=[[OUTPUT1]],[[OUTPUT2]]"

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-LIB: 2: append-footer, {1}, c++, (host-sycl)
// CHK-PHASES-LIB: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-LIB: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LIB: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-LIB: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {6}, c++-cpp-output
// CHK-PHASES-LIB: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASES-LIB: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASES-LIB: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASES-LIB: 11: linker, {0, 10}, image, (host-sycl)
// CHK-PHASES-LIB: 12: linker, {6}, ir, (device-sycl)
// CHK-PHASES-LIB: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASES-LIB: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASES-LIB: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-LIB: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {17}, image

/// Compilation check with -lstdc++ (treated differently than regular lib)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -lstdc++ -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LIB-STDCXX %s
// CHK-LIB-STDCXX: ld{{.*}} "-lstdc++"
// CHK-LIB-STDCXX-NOT: clang-offload-bundler{{.*}}
// CHK-LIB-STDCXX-NOT: llvm-link{{.*}} "-lstdc++"

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -fno-sycl-device-lib=all %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s
// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 2: append-footer, {1}, c++, (host-sycl)
// CHK-PHASES-FILES: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 4: input, "[[INPUT1]]", c++, (device-sycl)
// CHK-PHASES-FILES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-FILES: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {6}, c++-cpp-output
// CHK-PHASES-FILES: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASES-FILES: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASES-FILES: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASES-FILES: 11: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 12: append-footer, {11}, c++, (host-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 14: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-PHASES-FILES: 15: preprocessor, {14}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (device-sycl)
// CHK-PHASES-FILES: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, c++-cpp-output
// CHK-PHASES-FILES: 18: compiler, {17}, ir, (host-sycl)
// CHK-PHASES-FILES: 19: backend, {18}, assembler, (host-sycl)
// CHK-PHASES-FILES: 20: assembler, {19}, object, (host-sycl)
// CHK-PHASES-FILES: 21: linker, {0, 10, 20}, image, (host-sycl)
// CHK-PHASES-FILES: 22: linker, {6, 16}, ir, (device-sycl)
// CHK-PHASES-FILES: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASES-FILES: 26: file-table-tform, {23, 25}, tempfiletable, (device-sycl)
// CHK-PHASES-FILES: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHK-PHASES-FILES: 28: offload, "host-sycl (x86_64-unknown-linux-gnu)" {21}, "device-sycl (spir64-unknown-unknown-sycldevice)" {27}, image

/// ###########################################################################

/// Check separate compilation with offloading - bundling actions
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -c -o %t.o -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BUACTIONS %s
// CHK-BUACTIONS: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-BUACTIONS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-BUACTIONS: 2: compiler, {1}, ir, (device-sycl)
// CHK-BUACTIONS: 3: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {2}, ir
// CHK-BUACTIONS: 4: input, "[[INPUT]]", c++, (host-sycl)
// CHK-BUACTIONS: 5: append-footer, {4}, c++, (host-sycl)
// CHK-BUACTIONS: 6: preprocessor, {5}, c++-cpp-output, (host-sycl)
// CHK-BUACTIONS: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {6}, "device-sycl (spir64-unknown-unknown-sycldevice)" {2}, c++-cpp-output
// CHK-BUACTIONS: 8: compiler, {7}, ir, (host-sycl)
// CHK-BUACTIONS: 9: backend, {8}, assembler, (host-sycl)
// CHK-BUACTIONS: 10: assembler, {9}, object, (host-sycl)
// CHK-BUACTIONS: 11: clang-offload-bundler, {3, 10}, object, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64 %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown-sycldevice %t_dir/dummy 2>&1 \
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
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fno-sycl-device-lib=all %t.o -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-UBUACTIONS: 4: append-footer, {3}, c++, (host-sycl)
// CHK-UBUACTIONS: 5: preprocessor, {4}, c++-cpp-output, (host-sycl)
// CHK-UBUACTIONS: 6: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-UBUACTIONS: 7: preprocessor, {6}, c++-cpp-output, (device-sycl)
// CHK-UBUACTIONS: 8: compiler, {7}, ir, (device-sycl)
// CHK-UBUACTIONS: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {5}, "device-sycl (spir64-unknown-unknown-sycldevice)" {8}, c++-cpp-output
// CHK-UBUACTIONS: 10: compiler, {9}, ir, (host-sycl)
// CHK-UBUACTIONS: 11: backend, {10}, assembler, (host-sycl)
// CHK-UBUACTIONS: 12: assembler, {11}, object, (host-sycl)
// CHK-UBUACTIONS: 13: linker, {0, 2, 12}, image, (host-sycl)
// CHK-UBUACTIONS: 14: linker, {2, 8}, ir, (device-sycl)
// CHK-UBUACTIONS: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-UBUACTIONS: 20: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64-unknown-unknown-sycldevice)" {19}, image

/// ###########################################################################

/// Check -fsycl-is-device is passed when compiling for the device.
/// also check for SPIR-V binary creation
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-is-device and emitting to .spv when compiling for the device
/// when using -fno-sycl-use-bitcode
// RUN:   %clang -### -fno-sycl-use-bitcode -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s
// RUN:   %clang_cl -### -fno-sycl-use-bitcode -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s

// CHK-FSYCL-IS-DEVICE-NO-BITCODE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// CHK-LINK-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-TARGETS-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-TARGETS-UB: 3: llvm-spirv, {2}, image, (device-sycl)
// CHK-LINK-TARGETS-UB: 4: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {3}, image

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors unbundle multiple objects
// RUN:   touch %t-a.o
// RUN:   touch %t-b.o
// RUN:   touch %t-c.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// CHK-LINK-TARGETS-UB2: 0: input, "[[INPUT:.+\a.o]]", object
// CHK-LINK-TARGETS-UB2: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB2: 2: input, "[[INPUT:.+\b.o]]", object
// CHK-LINK-TARGETS-UB2: 3: clang-offload-unbundler, {2}, object
// CHK-LINK-TARGETS-UB2: 4: input, "[[INPUT:.+\c.o]]", object
// CHK-LINK-TARGETS-UB2: 5: clang-offload-unbundler, {4}, object
// CHK-LINK-TARGETS-UB2: 6: linker, {1, 3, 5}, image, (device-sycl)
// CHK-LINK-TARGETS-UB2: 7: llvm-spirv, {6}, image, (device-sycl)
// CHK-LINK-TARGETS-UB2: 8: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {7}, image

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s
// CHK-LINK-TARGETS: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-LINK-TARGETS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-LINK-TARGETS: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK-TARGETS: 3: linker, {2}, image, (device-sycl)
// CHK-LINK-TARGETS: 4: llvm-spirv, {3}, image, (device-sycl)
// CHK-LINK-TARGETS: 5: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, image

/// ###########################################################################

/// Check -fsycl-link behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// CHK-LINK-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-UB: 3: sycl-post-link, {2}, ir, (device-sycl)
// CHK-LINK-UB: 4: llvm-spirv, {3}, image, (device-sycl)
// CHK-LINK-UB: 5: clang-offload-wrapper, {4}, object, (device-sycl)
// CHK-LINK-UB: 6: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, object

/// ###########################################################################

/// Check -fsycl-link behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-LINK: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-LINK: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK: 3: linker, {2}, image, (device-sycl)
// CHK-LINK: 4: sycl-post-link, {3}, ir, (device-sycl)
// CHK-LINK: 5: llvm-spirv, {4}, image, (device-sycl)
// CHK-LINK: 6: clang-offload-wrapper, {5}, object, (device-sycl)
// CHK-LINK: 7: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {6}, object

/// ###########################################################################

/// Check -fsycl-add-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy.spv %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-UB %s
// CHK-ADD-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object, (host-sycl)
// CHK-ADD-TARGETS-UB: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-ADD-TARGETS-UB: 2: linker, {1}, image, (host-sycl)
// CHK-ADD-TARGETS-UB: 3: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-UB: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-ADD-TARGETS-UB: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, image

/// ###########################################################################

/// Check offload with multiple triples, multiple binaries passed through -fsycl-add-targets

// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy.spv,spir64_fpga-unknown-unknown-sycldevice:dummy.aocx,spir64_gen-unknown-unknown-sycldevice:dummy_Gen9core.bin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-MUL %s
// CHK-ADD-TARGETS-MUL: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-ADD-TARGETS-MUL: 1: append-footer, {0}, c++, (host-sycl)
// CHK-ADD-TARGETS-MUL: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-MUL: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-MUL: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-MUL: 5: compiler, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-MUL: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-ADD-TARGETS-MUL: 7: compiler, {6}, ir, (host-sycl)
// CHK-ADD-TARGETS-MUL: 8: backend, {7}, assembler, (host-sycl)
// CHK-ADD-TARGETS-MUL: 9: assembler, {8}, object, (host-sycl)
// CHK-ADD-TARGETS-MUL: 10: linker, {9}, image, (host-sycl)
// CHK-ADD-TARGETS-MUL: 11: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 12: clang-offload-wrapper, {11}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 13: input, "dummy.aocx", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 15: input, "dummy_Gen9core.bin", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {12}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {14}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {16}, image

/// ###########################################################################

/// Check offload with single triple, multiple binaries passed through -fsycl-add-targets

// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy0.spv,spir64-unknown-unknown-sycldevice:dummy1.spv,spir64-unknown-unknown-sycldevice:dummy2.spv %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-MUL-BINS %s
// CHK-ADD-TARGETS-MUL-BINS: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 1: append-footer, {0}, c++, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 5: compiler, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-ADD-TARGETS-MUL-BINS: 7: compiler, {6}, ir, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 8: backend, {7}, assembler, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 9: assembler, {8}, object, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 10: linker, {9}, image, (host-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 11: input, "dummy0.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 12: clang-offload-wrapper, {11}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 13: input, "dummy1.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 15: input, "dummy2.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-ADD-TARGETS-MUL-BINS: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {12}, "device-sycl (spir64-unknown-unknown-sycldevice)" {14}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, image

/// ###########################################################################

/// Check regular offload with an additional AOT binary passed through -fsycl-add-targets (same triple)

// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown-sycldevice -fsycl-add-targets=spir64-unknown-unknown-sycldevice:dummy.spv -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-REG %s
// CHK-ADD-TARGETS-REG: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-ADD-TARGETS-REG: 1: append-footer, {0}, c++, (host-sycl)
// CHK-ADD-TARGETS-REG: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-REG: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-REG: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-REG: 5: compiler, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-ADD-TARGETS-REG: 7: compiler, {6}, ir, (host-sycl)
// CHK-ADD-TARGETS-REG: 8: backend, {7}, assembler, (host-sycl)
// CHK-ADD-TARGETS-REG: 9: assembler, {8}, object, (host-sycl)
// CHK-ADD-TARGETS-REG: 10: linker, {9}, image, (host-sycl)
// CHK-ADD-TARGETS-REG: 11: linker, {5}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG: 15: file-table-tform, {12, 14}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-ADD-TARGETS-REG: 17: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-ADD-TARGETS-REG: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {16}, "device-sycl (spir64-unknown-unknown-sycldevice)" {18}, image

/// ###########################################################################

/// Check regular offload with multiple additional AOT binaries passed through -fsycl-add-targets
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown-sycldevice -fsycl-add-targets=spir64_fpga-unknown-unknown-sycldevice:dummy.aocx,spir64_gen-unknown-unknown-sycldevice:dummy_Gen9core.bin,spir64_x86_64-unknown-unknown-sycldevice:dummy.ir -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-REG-MUL %s
// CHK-ADD-TARGETS-REG-MUL: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 1: append-footer, {0}, c++, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 5: compiler, {4}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-ADD-TARGETS-REG-MUL: 7: compiler, {6}, ir, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 8: backend, {7}, assembler, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 9: assembler, {8}, object, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 10: linker, {9}, image, (host-sycl)
// CHK-ADD-TARGETS-REG-MUL: 11: input, "[[INPUT]]", c++, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 13: compiler, {12}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 14: linker, {13}, ir, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 20: input, "dummy.aocx", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 21: clang-offload-wrapper, {20}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 22: input, "dummy_Gen9core.bin", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 23: clang-offload-wrapper, {22}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 24: input, "dummy.ir", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 25: clang-offload-wrapper, {24}, object, (device-sycl)
// CHK-ADD-TARGETS-REG-MUL: 26: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {19}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {21}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {23}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {25}, image

/// ###########################################################################

/// Check for default linking of -lsycl with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-SYCL %s
// CHECK-LD-SYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SYCL: "-lsycl"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl -nolibsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOLIBSYCL %s
// CHECK-LD-NOLIBSYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOLIBSYCL-NOT: "-lsycl"

/// Check for default linking of sycl.lib with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// RUN: %clang_cl -fsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-CL %s
// CHECK-LINK-SYCL-CL: "--dependent-lib=sycl"
// CHECK-LINK-SYCL-CL-NOT: "-defaultlib:sycl.lib"
// CHECK-LINK-SYCL: "-defaultlib:sycl.lib"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl -nolibsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL %s
// RUN: %clang_cl -fsycl -nolibsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL %s
// CHECK-LINK-NOLIBSYCL-NOT: "--dependent-lib=sycl"
// CHECK-LINK-NOLIBSYCL: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOLIBSYCL-NOT: "-defaultlib:sycl.lib"

/// Check sycld.lib is chosen with /MDd
// RUN:  %clang_cl -fsycl /MDd %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
// CHECK-LINK-SYCL-DEBUG: "--dependent-lib=sycld"
// CHECK-LINK-SYCL-DEBUG-NOT: "-defaultlib:sycld.lib"

/// Check "-spirv-allow-unknown-intrinsics=llvm.genx." option is emitted for llvm-spirv tool
// RUN: %clangxx %s -fsycl -### 2>&1 | FileCheck %s --check-prefix=CHK-ALLOW-INTRIN
// CHK-ALLOW-INTRIN: llvm-spirv{{.*}}-spirv-allow-unknown-intrinsics=llvm.genx.

/// ###########################################################################

/// Check -Xsycl-target-backend triggers error when multiple triples are used.
// RUN:   %clang -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice,spir_fpga-unknown-unknown-sycldevice -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice,spir_fpga-unknown-unknown-sycldevice -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// CHK-FSYCL-TARGET-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-backend', specify triple using '-Xsycl-target-backend=<triple>'

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_gen %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64 %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-AOT: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-AOT: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-AOT: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-AOT: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-AOT: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FPGA: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-GEN: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-CPU: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-AOT: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-AOT: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-AOT: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-AOT: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES-AOT: 11: linker, {5}, ir, (device-sycl)
// CHK-PHASES-AOT: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-PHASES-AOT: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-PHASES-FPGA: 15: backend-compiler, {14}, fpga_aocx, (device-sycl)
// CHK-PHASES-GEN: 15: backend-compiler, {14}, image, (device-sycl)
// CHK-PHASES-CPU: 15: backend-compiler, {14}, image, (device-sycl)
// CHK-PHASES-AOT: 16: file-table-tform, {12, 15}, tempfiletable, (device-sycl)
// CHK-PHASES-AOT: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-FPGA: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {17}, image
// CHK-PHASES-GEN: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {17}, image
// CHK-PHASES-CPU: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_x86_64-unknown-unknown-sycldevice)" {17}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -Xssimulation %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-HW
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fintelfpga -Xsemulator %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA,CHK-TOOLS-FPGA-EMU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown-sycldevice"
// CHK-TOOLS-GEN: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice"
// CHK-TOOLS-CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown-sycldevice"
// CHK-TOOLS-AOT: "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOTx: "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-AOT: sycl-post-link{{.*}} "-o" "[[OUTPUT2_T:.+\.table]]" "[[OUTPUT2]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-extract=Code" "-drop_titles" "-o" "[[OUTPUT2_1:.+\.txt]]" "[[OUTPUT2_T]]"
// CHK-TOOLS-CPU: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=ocl-100" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-GEN: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=ocl-100" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-HW: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=legacy" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-EMU: llvm-spirv{{.*}} "-o" "[[OUTPUT3_T:.+\.txt]]" "-spirv-max-version=1.3" "-spirv-debug-info-version=ocl-100" "-spirv-allow-extra-diexpressions" "-spirv-allow-unknown-intrinsics=llvm.genx." {{.*}} "[[OUTPUT2_1]]"
// CHK-TOOLS-FPGA-HW: aoc{{.*}} "-o" "[[OUTPUT4_T:.+\.aocx]]" "[[OUTPUT3_T]]"
// CHK-TOOLS-FPGA-EMU: opencl-aot{{.*}} "-spv=[[OUTPUT3_T]]" "-ir=[[OUTPUT4_T:.+\.aocx]]"
// CHK-TOOLS-GEN: ocloc{{.*}} "-output" "[[OUTPUT4_T:.+\.out]]" {{.*}} "[[OUTPUT3_T]]"
// CHK-TOOLS-CPU: opencl-aot{{.*}} "-o=[[OUTPUT4_T:.+\.out]]" {{.*}} "[[OUTPUT3_T]]"
// CHK-TOOLS-AOT: file-table-tform{{.*}} "-o" "[[OUTPUT4:.+\.table]]" "{{.*}}.table"{{.*}} "[[OUTPUT4_T]]"
// CHK-TOOLS-FPGA: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_fpga{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-GEN: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_gen{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-CPU: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64_x86_64{{.*}}" "-kind=sycl" "-batch" "[[OUTPUT4]]"
// CHK-TOOLS-AOT: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT6:.+\.o]]" "[[OUTPUT5]]"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT7:.+\.o]]
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT7]]" "[[OUTPUT6]]" {{.*}} "-lsycl"

// Check to be sure that for windows, the 'exe' tools are called
// RUN: %clang_cl -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-CPU-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-CPU-WIN
// RUN: %clang_cl -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-GEN-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-GEN-WIN
// RUN: %clang_cl -fsycl -Xshardware -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-FPGA-WIN
// RUN: %clang -target x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-FPGA-WIN
// CHK-TOOLS-GEN-WIN: ocloc.exe{{.*}}
// CHK-TOOLS-CPU-WIN: opencl-aot.exe{{.*}}
// CHK-TOOLS-FPGA-WIN: aoc.exe{{.*}}

/// ###########################################################################

/// Check -Xsycl-target-backend option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xshardware -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
/// Check -Xs option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -XsDFOO1 -XsDFOO2 -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xs "-DFOO1 -DFOO2" -Xshardware %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// CHK-TOOLS-FPGA-OPTS: aoc{{.*}} "-o" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-FPGA-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-GEN-OPTS %s
// CHK-TOOLS-GEN-OPTS: ocloc{{.*}} "-output" {{.*}} "-output_no_suffix" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-GEN-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS %s
// CHK-TOOLS-CPU-OPTS: opencl-aot{{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-CPU-OPTS-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xsycl-target-backend "--bo='\"-DFOO1 -DFOO2\"'" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS3 %s
// CHK-TOOLS-CPU-OPTS3: opencl-aot{{.*}} "--bo=\"-DFOO1 -DFOO2\""

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS %s
// CHK-TOOLS-OPTS: clang-offload-wrapper{{.*}} "-compile-opts=-DFOO1 -DFOO2"

/// Check for implied options (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: clang-offload-wrapper{{.*}} "-compile-opts=-g -cl-opt-disable -DFOO1 -DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-FPGA %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-FPGA %s
// CHK-TOOLS-IMPLIED-OPTS-FPGA: opencl-aot{{.*}} "--bo=-g -cl-opt-disable" "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-CPU %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-CPU %s
// CHK-TOOLS-IMPLIED-OPTS-CPU: opencl-aot{{.*}} "--bo=-g -cl-opt-disable" "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-GEN %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-GEN %s
// CHK-TOOLS-IMPLIED-OPTS-GEN: ocloc{{.*}} "-options" "-g -cl-opt-disable" "-DFOO1" "-DFOO2"

/// Check -Xsycl-target-linker option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -Xshardware -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS2 %s
// CHK-TOOLS-FPGA-OPTS2: aoc{{.*}} "-o" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-FPGA-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-GEN-OPTS2 %s
// CHK-TOOLS-GEN-OPTS2: ocloc{{.*}} "-output" {{.*}} "-output_no_suffix" {{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-GEN-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts={{.*}}

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS2 %s
// CHK-TOOLS-CPU-OPTS2: opencl-aot{{.*}} "-DFOO1" "-DFOO2"
// CHK-TOOLS-CPU-OPTS2-NOT: clang-offload-wrapper{{.*}} "-link-opts=-DFOO1 -DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS2 %s
// CHK-TOOLS-OPTS2: clang-offload-wrapper{{.*}} "-link-opts=-DFOO1 -DFOO2"

// Sane-check "-compile-opts" and "-link-opts" passing for multiple targets
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice \
// RUN:   -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice "-device skl -cl-opt-disable" -Xsycl-target-linker=spir64-unknown-unknown-sycldevice "-cl-denorms-are-zero" %s 2>&1 \
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
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64_fpga-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASE-MULTI-TARG: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// CHK-PHASE-MULTI-TARG: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG: 10: linker, {9}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG: 11: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 13: compiler, {12}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 14: linker, {13}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: preprocessor, {20}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: compiler, {21}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: linker, {22}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: sycl-post-link, {23}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: file-table-tform, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: llvm-spirv, {25}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: backend-compiler, {26}, fpga_aocx, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: file-table-tform, {24, 27}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: clang-offload-wrapper, {28}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 30: linker, {5}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: sycl-post-link, {30}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 32: file-table-tform, {31}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 33: llvm-spirv, {32}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG: 34: backend-compiler, {33}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG: 35: file-table-tform, {31, 34}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 37: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {19}, "device-sycl (spir64_fpga-unknown-unknown-sycldevice)" {29}, "device-sycl (spir64_gen-unknown-unknown-sycldevice)" {36}, image

/// ###########################################################################
/// Verify that -save-temps does not crash
// RUN: %clang -fsycl -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1
// RUN: %clang -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown-sycldevice -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown-sycldevice -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-SAVE-TEMPS,CHK-FSYCL-SAVE-TEMPS-CONFL
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-fsycl-is-device"{{.*}} "-o" "[[DEVICE_BASE_NAME:[a-z0-9-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[HEADER_NAME:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[DEVICE_BASE_NAME]].bc"{{.*}} "[[DEVICE_BASE_NAME]].ii"
// CHK-FSYCL-SAVE-TEMPS: llvm-link{{.*}} "[[DEVICE_BASE_NAME]].bc"{{.*}} "-o" "[[LINKED_DEVICE_BC:.*\.bc]]"
// CHK-FSYCL-SAVE-TEMPS-CONFL-NOT: "[[DEVICE_BASE_NAME]].bc"{{.*}} "[[DEVICE_BASE_NAME]].bc"
// CHK-FSYCL-SAVE-TEMPS: sycl-post-link{{.*}} "-o" "[[DEVICE_BASE_NAME]].table" "[[LINKED_DEVICE_BC]]"
// CHK-FSYCL-SAVE-TEMPS: file-table-tform{{.*}} "-o" "[[DEVICE_BASE_NAME]].txt" "[[DEVICE_BASE_NAME]].table"
// CHK-FSYCL-SAVE-TEMPS: llvm-foreach{{.*}}llvm-spirv{{.*}} "-o" "[[SPIRV_FILE_LIST:.*\.txt]]" {{.*}}"[[DEVICE_BASE_NAME]].txt"
// CHK-FSYCL-SAVE-TEMPS-CONFL-NOT: "-o" "[[DEVICE_BASE_NAME]].txt" {{.*}}"[[DEVICE_BASE_NAME]].txt"
// CHK-FSYCL-SAVE-TEMPS: file-table-tform{{.*}} "-o" "[[PRE_WRAPPER_TABLE:.*\.table]]" "[[DEVICE_BASE_NAME]].table" "[[SPIRV_FILE_LIST]]"
// CHK-FSYCL-SAVE-TEMPS-CONFL-NOT: "-o" "[[DEVICE_BASE_NAME]].table"{{.*}} "[[DEVICE_BASE_NAME]].table"
// CHK-FSYCL-SAVE-TEMPS: clang-offload-wrapper{{.*}} "-o=[[WRAPPER_TEMPFILE_NAME:.+]].bc"{{.*}} "-batch" "[[PRE_WRAPPER_TABLE]]"
// CHK-FSYCL-SAVE-TEMPS: llc{{.*}} "-o" "[[DEVICE_OBJ_NAME:.*\.o]]"{{.*}} "[[WRAPPER_TEMPFILE_NAME]].bc"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-include" "[[HEADER_NAME]]"{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_BASE_NAME:[a-z0-9_-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].bc"{{.*}} "[[HOST_BASE_NAME:[a-z0-9_-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].s"{{.*}} "[[HOST_BASE_NAME]].bc"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].o"{{.*}} "[[HOST_BASE_NAME]].s"
// CHK-FSYCL-SAVE-TEMPS: ld{{.*}} "[[HOST_BASE_NAME]].o"{{.*}} "[[DEVICE_OBJ_NAME]]"

/// -fsycl with /Fo testing
// RUN: %clang_cl -fsycl /Fosomefile.obj -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=FO-CHECK %s
// FO-CHECK: clang{{.*}} "-fsycl-int-header=[[HEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// FO-CHECK: clang{{.*}} "-include" "[[HEADER]]" {{.*}} "-o" "[[OUTPUT2:.+\.obj]]"
// FO-CHECK: clang-offload-bundler{{.*}} "-outputs=somefile.obj" "-inputs=[[OUTPUT1]],[[OUTPUT2]]"

/// passing of a library should not trigger the unbundler
// RUN: touch %t.a
// RUN: touch %t.lib
// RUN: %clang -ccc-print-phases -fsycl -fno-sycl-device-lib=all %t.a %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// RUN: %clang_cl -ccc-print-phases -fsycl -fno-sycl-device-lib=all %t.lib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// LIB-UNBUNDLE-CHECK-NOT: clang-offload-unbundler

/// Options should not be duplicated in AOT calls
// RUN: %clang -fsycl -### -fsycl-targets=spir64_fpga -Xshardware -Xsycl-target-backend "-DBLAH" %s 2>&1 \
// RUN:  | FileCheck -check-prefix=DUP-OPT %s
// DUP-OPT-NOT: aoc{{.*}} "-DBLAH" {{.*}} "-DBLAH"

/// passing of only a library should not create a device link
// RUN: %clang -ccc-print-phases -fsycl -lsomelib 2>&1 \
// RUN:  | FileCheck -check-prefix=LIB-NODEVICE %s
// LIB-NODEVICE: 0: input, "somelib", object, (host-sycl)
// LIB-NODEVICE: 1: linker, {0}, image, (host-sycl)
// LIB-NODEVICE-NOT: linker, {{.*}}, spirv, (device-sycl)

// Checking for an error if c-compilation is forced
// RUN: %clangxx -### -c -fsycl -xc %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: %clangxx -### -c -fsycl -xc-header %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: %clangxx -### -c -fsycl -xcpp-output %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// CHECK_XC_FSYCL: '-x c{{.*}}' must not be used in conjunction with '-fsycl'

// -std=c++17 check (check all 3 compilations)
// RUN: %clangxx -### -c -fsycl -xc++ %s 2>&1 | FileCheck -check-prefix=CHECK-STD %s
// RUN: %clang_cl -### -c -fsycl -TP %s 2>&1 | FileCheck -check-prefix=CHECK-STD %s
// CHECK-STD: clang{{.*}} "-emit-llvm-bc" {{.*}} "-std=c++17"
// CHECK-STD: clang{{.*}} "-emit-obj" {{.*}} "-std=c++17"

// -std=c++17 override check
// RUN: %clangxx -### -c -fsycl -std=c++14 -xc++ %s 2>&1 | FileCheck -check-prefix=CHECK-STD-OVR %s
// RUN: %clang_cl -### -c -fsycl /std:c++14 -TP %s 2>&1 | FileCheck -check-prefix=CHECK-STD-OVR %s
// CHECK-STD-OVR: clang{{.*}} "-emit-llvm-bc" {{.*}} "-std=c++14"
// CHECK-STD-OVR: clang{{.*}} "-emit-obj" {{.*}} "-std=c++14"
// CHECK-STD-OVR-NOT: clang{{.*}} "-std=c++17"

// Check sycl-post-link optimization level.
// Default is O2
// RUN:   %clang    -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// RUN:   %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// Common options for %clang and %clang_cl
// RUN:   %clang    -### -fsycl -O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// RUN:   %clang_cl -### -fsycl /O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// RUN:   %clang    -### -fsycl -O2 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// RUN:   %clang_cl -### -fsycl /O2 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// RUN:   %clang    -### -fsycl -Os %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// RUN:   %clang_cl -### -fsycl /Os %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// %clang options
// RUN:   %clang    -### -fsycl -O0 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O0
// RUN:   %clang    -### -fsycl -Ofast %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// RUN:   %clang    -### -fsycl -O3 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// RUN:   %clang    -### -fsycl -Oz %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Oz
// RUN:   %clang    -### -fsycl -Og %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// %clang_cl options
// RUN:   %clang_cl -### -fsycl /Od %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O0
// RUN:   %clang_cl -### -fsycl /Ot %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// only the last option is considered
// RUN:   %clang    -### -fsycl -O2 -O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// RUN:   %clang_cl -### -fsycl /O2 /O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// CHK-POST-LINK-OPT-LEVEL-O0: sycl-post-link{{.*}} "-O0"
// CHK-POST-LINK-OPT-LEVEL-O1: sycl-post-link{{.*}} "-O1"
// CHK-POST-LINK-OPT-LEVEL-O2: sycl-post-link{{.*}} "-O2"
// CHK-POST-LINK-OPT-LEVEL-O3: sycl-post-link{{.*}} "-O3"
// CHK-POST-LINK-OPT-LEVEL-Os: sycl-post-link{{.*}} "-Os"
// CHK-POST-LINK-OPT-LEVEL-Oz: sycl-post-link{{.*}} "-Oz"

// Verify header search dirs are added with -fsycl
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// RUN: %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-device"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl" "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-host"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl" "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"{{.*}}

/// Check for an incompatibility error when -fsycl and -ffreestanding used
// RUN:   %clang -### -fsycl -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s
// CHK-INCOMPATIBILITY: error: The option -fsycl conflicts with -ffreestanding
