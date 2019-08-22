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
// RUN:   %clang -### -fsycl-targets=spir64-unknown-linux-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   %clang_cl -### -fsycl-targets=spir64-unknown-linux-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: The option -fsycl-targets must be used in conjunction with -fsycl to enable offloading.

/// ###########################################################################

/// Check error for -fsycl-add-targets -fsycl-link-targets conflict
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-linux-sycldevice -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-linux-sycldevice -fsycl-add-targets=spir64:dummy.spv -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-ADD-LINK %s
// CHK-SYCL-ADD-LINK: error: The option -fsycl-link-targets= conflicts with -fsycl-add-targets=

/// ###########################################################################

/// Check error for -fsycl-targets -fsycl-link-targets conflict
// RUN:   %clang -### -fsycl-link-targets=spir64-unknown-linux-sycldevice -fsycl-targets=spir64-unknown-linux-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// RUN:   %clang_cl -### -fsycl-link-targets=spir64-unknown-linux-sycldevice -fsycl-targets=spir64-unknown-linux-sycldevice -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-LINK-CONFLICT %s
// CHK-SYCL-LINK-CONFLICT: error: The option -fsycl-targets= conflicts with -fsycl-link-targets=

/// ###########################################################################

/// Check error for -fsycl-targets -fintelfpga conflict
// RUN:   %clang -### -fsycl-targets=spir64-unknown-linux-sycldevice -fintelfpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// RUN:   %clang_cl -### -fsycl-targets=spir64-unknown-linux-sycldevice -fintelfpga -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-FPGA-CONFLICT %s
// CHK-SYCL-FPGA-CONFLICT: error: The option -fsycl-targets= conflicts with -fintelfpga

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir64-unknown-linux-sycldevice  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: The SYCL offloading target 'spir64-unknown-linux-sycldevice' is similar to target 'spir64-unknown-linux-sycldevice' already specified - will be ignored.

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when multiple triples are used.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir-unknown-linux-sycldevice -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir-unknown-linux-sycldevice -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when an option requiring arguments is passed to it.
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// CHK-FSYCL-COMPILER-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target-frontend argument: '-Xsycl-target-frontend -Xsycl-target-frontend', options requiring arguments are unsupported

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-use-bitcode
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-use-bitcode %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASES: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-{{.*}}-sycldevice)" {4}, cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES: 11: backend, {10}, assembler, (device-sycl)
// CHK-PHASES: 12: assembler, {11}, object, (device-sycl)
// CHK-PHASES: 13: linker, {12}, spirv, (device-sycl)
// CHK-PHASES: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-PHASES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-linux-sycldevice)" {14}, image

/// ###########################################################################

/// Check the compilation flow to verify that the integrated header is filtered
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -c %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHK-INT-HEADER
// CHK-INT-HEADER: clang{{.*}} "-fsycl-is-device" {{.*}} "-o" "[[OUTPUT1:.+\.o]]"
// CHK-INT-HEADER: clang{{.*}} "-triple" "spir64-unknown-{{windows|linux}}-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-INT-HEADER: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" "-dependency-filter" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT2:.+.o]]"
// CHK-INT-HEADER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-{{windows|linux}}-sycldevice,host-x86_64-unknown-linux-gnu" {{.*}} "-inputs=[[OUTPUT1]],[[OUTPUT2]]"

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES-LIB: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-LIB: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-LIB: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-linux-sycldevice)" {5}, cpp-output
// CHK-PHASES-LIB: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-LIB: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-LIB: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-LIB: 10: linker, {0, 9}, image, (host-sycl)
// CHK-PHASES-LIB: 11: input, "somelib", object, (device-sycl)
// CHK-PHASES-LIB: 12: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LIB: 13: backend, {12}, assembler, (device-sycl)
// CHK-PHASES-LIB: 14: assembler, {13}, object, (device-sycl)
// CHK-PHASES-LIB: 15: linker, {11, 14}, spirv, (device-sycl)
// CHK-PHASES-LIB: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASES-LIB: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-linux-sycldevice)" {16}, image

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s

// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: input, "[[INPUT1]]", c, (device-sycl)
// CHK-PHASES-FILES: 4: preprocessor, {3}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 5: compiler, {4}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-linux-sycldevice)" {5}, cpp-output
// CHK-PHASES-FILES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-FILES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-FILES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-FILES: 10: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-PHASES-FILES: 11: preprocessor, {10}, cpp-output, (host-sycl)
// CHK-PHASES-FILES: 12: input, "[[INPUT2]]", c, (device-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, cpp-output, (device-sycl)
// CHK-PHASES-FILES: 14: compiler, {13}, sycl-header, (device-sycl)
// CHK-PHASES-FILES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-linux-sycldevice)" {14}, cpp-output
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (host-sycl)
// CHK-PHASES-FILES: 17: backend, {16}, assembler, (host-sycl)
// CHK-PHASES-FILES: 18: assembler, {17}, object, (host-sycl)
// CHK-PHASES-FILES: 19: linker, {0, 9, 18}, image, (host-sycl)
// CHK-PHASES-FILES: 20: input, "somelib", object, (device-sycl)
// CHK-PHASES-FILES: 21: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-FILES: 22: backend, {21}, assembler, (device-sycl)
// CHK-PHASES-FILES: 23: assembler, {22}, object, (device-sycl)
// CHK-PHASES-FILES: 24: compiler, {13}, ir, (device-sycl)
// CHK-PHASES-FILES: 25: backend, {24}, assembler, (device-sycl)
// CHK-PHASES-FILES: 26: assembler, {25}, object, (device-sycl)
// CHK-PHASES-FILES: 27: linker, {20, 23, 26}, spirv, (device-sycl)
// CHK-PHASES-FILES: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASES-FILES: 29: offload, "host-sycl (x86_64-unknown-linux-gnu)" {19}, "device-sycl (spir64-unknown-linux-sycldevice)" {28}, image

/// ###########################################################################

/// Check separate compilation with offloading - bundling actions
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -c -o %t.o -lsomelib -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BUACTIONS %s
// CHK-BUACTIONS: 0: input, "[[INPUT:.+\.c]]", c, (device-sycl)
// CHK-BUACTIONS: 1: preprocessor, {0}, cpp-output, (device-sycl)
// CHK-BUACTIONS: 2: compiler, {1}, ir, (device-sycl)
// CHK-BUACTIONS: 3: backend, {2}, assembler, (device-sycl)
// CHK-BUACTIONS: 4: assembler, {3}, object, (device-sycl)
// CHK-BUACTIONS: 5: offload, "device-sycl (spir64-unknown-linux-sycldevice)" {4}, object
// CHK-BUACTIONS: 6: input, "[[INPUT]]", c, (host-sycl)
// CHK-BUACTIONS: 7: preprocessor, {6}, cpp-output, (host-sycl)
// CHK-BUACTIONS: 8: compiler, {1}, sycl-header, (device-sycl)
// CHK-BUACTIONS: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {7}, "device-sycl (spir64-unknown-linux-sycldevice)" {8}, cpp-output
// CHK-BUACTIONS: 10: compiler, {9}, ir, (host-sycl)
// CHK-BUACTIONS: 11: backend, {10}, assembler, (host-sycl)
// CHK-BUACTIONS: 12: assembler, {11}, object, (host-sycl)
// CHK-BUACTIONS: 13: clang-offload-bundler, {5, 12}, object, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -lsomelib -fsycl-targets=spir64-unknown-linux-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBACTIONS %s
// CHK-UBACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBACTIONS: 1: input, "[[INPUT:.+\.o]]", object, (host-sycl)
// CHK-UBACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBACTIONS: 3: linker, {0, 2}, image, (host-sycl)
// CHK-UBACTIONS: 4: input, "somelib", object, (device-sycl)
// CHK-UBACTIONS: 5: linker, {4, 2}, spirv, (device-sycl)
// CHK-UBACTIONS: 6: clang-offload-wrapper, {5}, object, (device-sycl)
// CHK-UBACTIONS: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-linux-sycldevice)" {6}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling with source
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl %t.o -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-UBUACTIONS %s
// CHK-UBUACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBUACTIONS: 1: input, "[[INPUT1:.+\.o]]", object, (host-sycl)
// CHK-UBUACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBUACTIONS: 3: input, "[[INPUT2:.+\.c]]", c, (host-sycl)
// CHK-UBUACTIONS: 4: preprocessor, {3}, cpp-output, (host-sycl)
// CHK-UBUACTIONS: 5: input, "[[INPUT2]]", c, (device-sycl)
// CHK-UBUACTIONS: 6: preprocessor, {5}, cpp-output, (device-sycl)
// CHK-UBUACTIONS: 7: compiler, {6}, sycl-header, (device-sycl)
// CHK-UBUACTIONS: 8: offload, "host-sycl (x86_64-unknown-linux-gnu)" {4}, "device-sycl (spir64-unknown-linux-sycldevice)" {7}, cpp-output
// CHK-UBUACTIONS: 9: compiler, {8}, ir, (host-sycl)
// CHK-UBUACTIONS: 10: backend, {9}, assembler, (host-sycl)
// CHK-UBUACTIONS: 11: assembler, {10}, object, (host-sycl)
// CHK-UBUACTIONS: 12: linker, {0, 2, 11}, image, (host-sycl)
// CHK-UBUACTIONS: 13: input, "somelib", object, (device-sycl)
// CHK-UBUACTIONS: 14: compiler, {6}, ir, (device-sycl)
// CHK-UBUACTIONS: 15: backend, {14}, assembler, (device-sycl)
// CHK-UBUACTIONS: 16: assembler, {15}, object, (device-sycl)
// CHK-UBUACTIONS: 17: linker, {13, 2, 16}, spirv, (device-sycl)
// CHK-UBUACTIONS: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-UBUACTIONS: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (spir64-unknown-linux-sycldevice)" {18}, image

/// ###########################################################################

/// Check -fsycl-is-device is passed when compiling for the device.
/// also check for SPIR-V binary creation
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-windows-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-is-device and emitting to .spv when compiling for the device
/// when using -fno-sycl-use-bitcode
// RUN:   %clang -### -fno-sycl-use-bitcode -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s
// RUN:   %clang_cl -### -fno-sycl-use-bitcode -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s

// CHK-FSYCL-IS-DEVICE-NO-BITCODE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB %s
// CHK-LINK-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-TARGETS-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-TARGETS-UB: 3: offload, "device-sycl (spir64-unknown-linux-sycldevice)" {2}, image

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors unbundle multiple objects
// RUN:   touch %t-a.o
// RUN:   touch %t-b.o
// RUN:   touch %t-c.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %t-a.o %t-b.o %t-c.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS-UB2 %s
// CHK-LINK-TARGETS-UB2: 0: input, "[[INPUT:.+\a.o]]", object
// CHK-LINK-TARGETS-UB2: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-TARGETS-UB2: 2: input, "[[INPUT:.+\b.o]]", object
// CHK-LINK-TARGETS-UB2: 3: clang-offload-unbundler, {2}, object
// CHK-LINK-TARGETS-UB2: 4: input, "[[INPUT:.+\c.o]]", object
// CHK-LINK-TARGETS-UB2: 5: clang-offload-unbundler, {4}, object
// CHK-LINK-TARGETS-UB2: 6: linker, {1, 3, 5}, image, (device-sycl)
// CHK-LINK-TARGETS-UB2: 7: offload, "device-sycl (spir64-unknown-linux-sycldevice)" {6}, image

/// ###########################################################################

/// Check -fsycl-link-targets=<triple> behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link-targets=spir64-unknown-linux-sycldevice %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-TARGETS %s
// CHK-LINK-TARGETS: 0: input, "[[INPUT:.+\.c]]", c, (device-sycl)
// CHK-LINK-TARGETS: 1: preprocessor, {0}, cpp-output, (device-sycl)
// CHK-LINK-TARGETS: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK-TARGETS: 3: backend, {2}, assembler, (device-sycl)
// CHK-LINK-TARGETS: 4: assembler, {3}, object, (device-sycl)
// CHK-LINK-TARGETS: 5: linker, {4}, image, (device-sycl)
// CHK-LINK-TARGETS: 6: offload, "device-sycl (spir64-unknown-{{linux|windows}}-sycldevice)" {5}, image

/// ###########################################################################

/// Check -fsycl-link behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// CHK-LINK-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-UB: 3: clang-offload-wrapper, {2}, object, (device-sycl)
// CHK-LINK-UB: 4: offload, "device-sycl (spir64-unknown-{{linux|windows}}-sycldevice{{.*}})" {3}, object

/// ###########################################################################

/// Check -fsycl-link behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// RUN:   %clang_cl -### -ccc-print-phases -fsycl -o %t.out -fsycl-link %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: 0: input, "[[INPUT:.+\.c]]", c, (device-sycl)
// CHK-LINK: 1: preprocessor, {0}, cpp-output, (device-sycl)
// CHK-LINK: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK: 3: backend, {2}, assembler, (device-sycl)
// CHK-LINK: 4: assembler, {3}, object, (device-sycl)
// CHK-LINK: 5: linker, {4}, image, (device-sycl)
// CHK-LINK: 6: clang-offload-wrapper, {5}, object, (device-sycl)
// CHK-LINK: 7: offload, "device-sycl (spir64-unknown-{{linux|windows}}-sycldevice{{.*}})" {6}, object

/// ###########################################################################

/// Check -fsycl-add-targets=<triple> behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-add-targets=spir64-unknown-linux-sycldevice:dummy.spv %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ADD-TARGETS-UB %s
// CHK-ADD-TARGETS-UB: 0: input, "[[INPUT:.+\.o]]", object, (host-sycl)
// CHK-ADD-TARGETS-UB: 1: clang-offload-unbundler, {0}, object, (host-sycl)
// CHK-ADD-TARGETS-UB: 2: linker, {1}, image, (host-sycl)
// CHK-ADD-TARGETS-UB: 3: input, "dummy.spv", sycl-fatbin, (device-sycl)
// CHK-ADD-TARGETS-UB: 4: clang-offload-wrapper, {3}, object, (device-sycl)
// CHK-ADD-TARGETS-UB: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-linux-sycldevice)" {4}, image

/// ###########################################################################

/// Check for default linking of -lsycl with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-SYCL %s
// CHECK-LD-SYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SYCL: "-lsycl"

/// Check for default linking of sycl.lib with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// RUN: %clang_cl -fsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// CHECK-LINK-SYCL: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-SYCL: "-defaultlib:sycl.lib"

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib>
// RUN: touch %t.a
// RUN: touch %t.o
// RUN: %clang -fsycl -L/dummy/dir -foffload-static-lib=%t.a -### %t.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// FOFFLOAD_STATIC_LIB: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.o]]" "-L/dummy/dir" "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB: llvm-link{{.*}} "@{{.*}}"

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib> with multiple objects
// RUN: touch %t.a
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: touch %t-3.o
// RUN: %clang -fsycl -foffload-static-lib=%t.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// FOFFLOAD_STATIC_LIB_MULTI_O: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\-1.o]]" "[[INPUT:.+\-2.o]]" "[[INPUT:.+\-3.o]]" "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{.*}} "@{{.*}}"

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib> from source
// RUN: touch %t.a
// RUN: %clang -fsycl -foffload-static-lib=%t.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: preprocessor, {0}, cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: input, "[[INPUT]]", c, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: preprocessor, {2}, cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: compiler, {3}, sycl-header, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-linux-sycldevice)" {4}, cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 6: compiler, {5}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 7: backend, {6}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 8: assembler, {7}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: clang-offload-unbundler, {8}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: linker, {9}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: compiler, {3}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: backend, {11}, assembler, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: assembler, {12}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 14: linker, {13, 9}, spirv, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-linux-sycldevice)" {15}, image

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clang -fsycl -foffload-static-lib=%t.a -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// FOFFLOAD_STATIC_LIB_SRC2: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{.*}} "@{{.*}}"

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clang -fsycl -foffload-static-lib=%t.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC3
// FOFFLOAD_STATIC_LIB_SRC3: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_SRC3: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_SRC3: llvm-link{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB_SRC3: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lOpenCL"

/// ###########################################################################

/// Check -Xsycl-target-backend triggers error when multiple triples are used.
// RUN:   %clang -### -fsycl -fsycl-targets=spir64_fpga-unknown-linux-sycldevice,spir_fpga-unknown-linux-sycldevice -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64_fpga-unknown-linux-sycldevice,spir_fpga-unknown-linux-sycldevice -Xsycl-target-backend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-TARGET-AMBIGUOUS-ERROR %s
// CHK-FSYCL-TARGET-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-backend', specify triple using '-Xsycl-target-backend=<triple>'

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-targets=spir64_fpga-unknown-linux-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-FPGA
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-targets=spir64_gen-unknown-linux-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-GEN
// RUN:   %clang -target x86_64-unknown-linux-gnu -ccc-print-phases -fsycl -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice %s 2>&1 \
// RUN:    | FileCheck %s -check-prefixes=CHK-PHASES-AOT,CHK-PHASES-CPU
// CHK-PHASES-AOT: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASES-AOT: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASES-AOT: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASES-AOT: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASES-AOT: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES-FPGA: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_fpga-unknown-linux-sycldevice)" {4}, cpp-output
// CHK-PHASES-GEN: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-linux-sycldevice)" {4}, cpp-output
// CHK-PHASES-CPU: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_x86_64-unknown-linux-sycldevice)" {4}, cpp-output
// CHK-PHASES-AOT: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-AOT: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-AOT: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-AOT: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-AOT: 10: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-AOT: 11: backend, {10}, assembler, (device-sycl)
// CHK-PHASES-AOT: 12: assembler, {11}, object, (device-sycl)
// CHK-PHASES-AOT: 13: linker, {12}, spirv, (device-sycl)
// CHK-PHASES-GEN: 14: backend-compiler, {13}, image, (device-sycl)
// CHK-PHASES-FPGA: 14: backend-compiler, {13}, fpga-aocx, (device-sycl)
// CHK-PHASES-AOT: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES-FPGA: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_fpga-unknown-linux-sycldevice)" {15}, image
// CHK-PHASES-GEN: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_gen-unknown-linux-sycldevice)" {15}, image
// CHK-PHASES-CPU: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64_x86_64-unknown-linux-sycldevice)" {15}, image

/// ###########################################################################

/// Ahead of Time compilation for fpga, gen, cpu - tool invocation
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-linux-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fintelfpga %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-FPGA
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-linux-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-GEN
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHK-TOOLS-AOT,CHK-TOOLS-CPU
// CHK-TOOLS-AOT: clang{{.*}} "-fsycl-is-device" {{.*}} "-o" "[[OUTPUT1:.+\.o]]"
// CHK-TOOLS-AOT: llvm-link{{.*}} "[[OUTPUT1]]" "-o" "[[OUTPUT2:.+\.bc]]"
// CHK-TOOLS-AOT: llvm-spirv{{.*}} "-o" "[[OUTPUT3:.+\.spv]]" "[[OUTPUT2]]"
// CHK-TOOLS-FPGA: aoc{{.*}} "-o" "[[OUTPUT4:.+\.aocx]]" "[[OUTPUT3]]"
// CHK-TOOLS-GEN: ocloc{{.*}} "-output" "[[OUTPUT4:.+\.out]]" {{.*}} "[[OUTPUT3]]"
// CHK-TOOLS-CPU: ioc{{.*}} "-ir=[[OUTPUT4:.+\.out]]" {{.*}} "-binary=[[OUTPUT3]]"
// CHK-TOOLS-AOT: clang-offload-wrapper{{.*}} "-o=[[OUTPUT5:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-kind=sycl" "[[OUTPUT4]]"
// CHK-TOOLS-AOT: llc{{.*}} "-filetype=obj" "-o" "[[OUTPUT6:.+\.o]]" "[[OUTPUT5]]"
// CHK-TOOLS-FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-{{.*}}-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-GEN: clang{{.*}} "-triple" "spir64_gen-unknown-linux-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-linux-sycldevice" {{.*}} "-fsycl-int-header=[[INPUT1:.+\.h]]" "-faddrsig"
// CHK-TOOLS-AOT: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT7:.+\.o]]"
// CHK-TOOLS-AOT: ld{{.*}} "[[OUTPUT7]]" "[[OUTPUT6]]" {{.*}} "-lsycl"

/// ###########################################################################

/// Check -Xsycl-target-backend option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-linux-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
/// Check -Xs option passing
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -XsDFOO1 -XsDFOO2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fintelfpga -Xs "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-FPGA-OPTS %s
// CHK-TOOLS-FPGA-OPTS: aoc{{.*}} "-o" {{.*}} "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-linux-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-GEN-OPTS %s
// CHK-TOOLS-GEN-OPTS: ocloc{{.*}} "-output" {{.*}} "-DFOO1" "-DFOO2"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-CPU-OPTS %s
// CHK-TOOLS-CPU-OPTS: ioc{{.*}} "-DFOO1" "-DFOO2"

/// ###########################################################################

/// offload with multiple targets, including AOT
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice,spir64_fpga-unknown-linux-sycldevice,spir64_gen-unknown-linux-sycldevice -###  -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG %s
// CHK-PHASE-MULTI-TARG: 0: input, "[[INPUT:.+\.c]]", c, (host-sycl)
// CHK-PHASE-MULTI-TARG: 1: preprocessor, {0}, cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG: 2: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 3: preprocessor, {2}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASE-MULTI-TARG: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-linux-sycldevice)" {4}, cpp-output
// CHK-PHASE-MULTI-TARG: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG: 9: linker, {8}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG: 10: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 11: preprocessor, {10}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 12: compiler, {11}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 13: backend, {12}, assembler, (device-sycl)
// CHK-PHASE-MULTI-TARG: 14: assembler, {13}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 15: linker, {14}, spirv, (device-sycl)
// CHK-PHASE-MULTI-TARG: 16: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 17: input, "[[INPUT]]", c, (device-sycl)
// CHK-PHASE-MULTI-TARG: 18: preprocessor, {17}, cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG: 19: compiler, {18}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 20: backend, {19}, assembler, (device-sycl)
// CHK-PHASE-MULTI-TARG: 21: assembler, {20}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 22: linker, {21}, spirv, (device-sycl)
// CHK-PHASE-MULTI-TARG: 23: backend-compiler, {22}, fpga-aocx, (device-sycl)
// CHK-PHASE-MULTI-TARG: 24: clang-offload-wrapper, {23}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 25: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG: 26: backend, {25}, assembler, (device-sycl)
// CHK-PHASE-MULTI-TARG: 27: assembler, {26}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 28: linker, {27}, spirv, (device-sycl)
// CHK-PHASE-MULTI-TARG: 29: backend-compiler, {28}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG: 30: clang-offload-wrapper, {29}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG: 31: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (spir64-unknown-linux-sycldevice)" {16}, "device-sycl (spir64_fpga-unknown-linux-sycldevice)" {24}, "device-sycl (spir64_gen-unknown-linux-sycldevice)" {30}, image

/// ###########################################################################
/// Verify that -save-temps does not crash
// RUN: %clang -fsycl -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1
// RUN: %clang -fsycl -fsycl-targets=spir64-unknown-linux-sycldevice -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
