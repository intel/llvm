///
/// Perform several driver tests for SYCL offloading
///

// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// RUN:   not %clang_cl -### -fsycl -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: SYCL target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// RUN:   not %clang_cl -### -fsycl -fsycl-targets=x86_64 %s 2>&1 \
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
// RUN:   not %clang -### -fsycl-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   not %clang_cl -### -fsycl-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-targets' must be used in conjunction with '-fsycl' to enable offloading
// RUN:   not %clang -### -fsycl-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-LINK %s
// CHK-NO-FSYCL-LINK: error: '-fsycl-link' must be used in conjunction with '-fsycl' to enable offloading

/// ###########################################################################

/// Validate SYCL option values
// RUN:   not %clang -### -fsycl-device-code-split=bad_value -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-device-code-split %s
// RUN:   not %clang -### -fsycl-link=bad_value -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-link %s
// CHK-SYCL-BAD-OPT-VALUE: error: invalid argument 'bad_value' to [[option]]=

/// Check no error for -fsycl-targets with good triple
// RUN:   %clang -### -fsycl-targets=spir-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64 -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang_cl -### -fsycl-targets=spir-unknown-unknown -fsycl  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// CHK-SYCL-TARGET-NOT: error: SYCL target is invalid

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl -fsycl-targets=spir64-unknown-unknown,spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: SYCL offloading target 'spir64-unknown-unknown' is similar to target 'spir64-unknown-unknown' already specified; will be ignored

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when multiple triples are used.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when an option requiring arguments is passed to it.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// CHK-FSYCL-COMPILER-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target-frontend argument: '-Xsycl-target-frontend -Xsycl-target-frontend', options requiring arguments are unsupported

/// ###########################################################################

/// Check -Xsycl-target-frontend= accepts triple aliases
// RUN:   %clang -### -fsycl -fsycl-targets=spir64 -Xsycl-target-frontend=spir64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH=spir64 -check-prefixes=CHK-UNUSED-ARG-WARNING,CHK-TARGET %s
// CHK-UNUSED-ARG-WARNING-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET: clang{{.*}} "-cc1" "-triple" "[[ARCH]]-unknown-unknown"{{.*}} "-D" "FOO"

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-device-obj=llvmir
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-obj=spirv -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fsycl-device-obj=spirv -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-device-obj=llvmir -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-DEFAULT-MODE %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -fsycl-device-obj=llvmir -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES,CHK-PHASES-CL-MODE %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-DEFAULT-MODE: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHK-PHASES-CL-MODE: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
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
// CHK-PHASES-DEFAULT-MODE: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown)" {16}, image
// CHK-PHASES-CL-MODE: 17: offload, "host-sycl (x86_64-pc-windows-msvc)" {10}, "device-sycl (spir64-unknown-unknown)" {16}, image

/// ###########################################################################

/// Check the compilation flow to verify that the integrated header is filtered
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl -c %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHK-INT-HEADER
// CHK-INT-HEADER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-INT-HEADER: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[INPUT1]]" "-dependency-filter" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT2:.+\.o]]"
// CHK-INT-HEADER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown,host-x86_64-unknown-linux-gnu" {{.*}} "-input=[[OUTPUT1]]" "-input=[[OUTPUT2]]"

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-LIB: 2: append-footer, {1}, c++, (host-sycl)
// CHK-PHASES-LIB: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-LIB: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LIB: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-LIB: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {6}, c++-cpp-output
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
// CHK-PHASES-LIB: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown)" {17}, image

/// Compilation check with -lstdc++ (treated differently than regular lib)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -lstdc++ -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LIB-STDCXX %s
// CHK-LIB-STDCXX: ld{{.*}} "-lstdc++"
// CHK-LIB-STDCXX-NOT: clang-offload-bundler{{.*}}
// CHK-LIB-STDCXX-NOT: llvm-link{{.*}} "-lstdc++"

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s
// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 2: append-footer, {1}, c++, (host-sycl)
// CHK-PHASES-FILES: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 4: input, "[[INPUT1]]", c++, (device-sycl)
// CHK-PHASES-FILES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-FILES: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {6}, c++-cpp-output
// CHK-PHASES-FILES: 8: compiler, {7}, ir, (host-sycl)
// CHK-PHASES-FILES: 9: backend, {8}, assembler, (host-sycl)
// CHK-PHASES-FILES: 10: assembler, {9}, object, (host-sycl)
// CHK-PHASES-FILES: 11: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 12: append-footer, {11}, c++, (host-sycl)
// CHK-PHASES-FILES: 13: preprocessor, {12}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 14: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-PHASES-FILES: 15: preprocessor, {14}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 16: compiler, {15}, ir, (device-sycl)
// CHK-PHASES-FILES: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64-unknown-unknown)" {16}, c++-cpp-output
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
// CHK-PHASES-FILES: 28: offload, "host-sycl (x86_64-unknown-linux-gnu)" {21}, "device-sycl (spir64-unknown-unknown)" {27}, image

/// ###########################################################################

/// Check separate compilation with offloading - bundling actions
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -c -o %t.o -lsomelib -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BUACTIONS %s
// CHK-BUACTIONS: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-BUACTIONS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-BUACTIONS: 2: compiler, {1}, ir, (device-sycl)
// CHK-BUACTIONS: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// CHK-BUACTIONS: 4: input, "[[INPUT]]", c++, (host-sycl)
// CHK-BUACTIONS: 5: append-footer, {4}, c++, (host-sycl)
// CHK-BUACTIONS: 6: preprocessor, {5}, c++-cpp-output, (host-sycl)
// CHK-BUACTIONS: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {6}, "device-sycl (spir64-unknown-unknown)" {2}, c++-cpp-output
// CHK-BUACTIONS: 8: compiler, {7}, ir, (host-sycl)
// CHK-BUACTIONS: 9: backend, {8}, assembler, (host-sycl)
// CHK-BUACTIONS: 10: assembler, {9}, object, (host-sycl)
// CHK-BUACTIONS: 11: clang-offload-bundler, {3, 10}, object, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64 %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t_dir/dummy 2>&1 \
// RUN:   | FileCheck -DINPUT=%t_dir/dummy -check-prefix=CHK-UBACTIONS %s
// CHK-UBACTIONS: 0: input, "somelib", object, (host-sycl)
// CHK-UBACTIONS: 1: input, "[[INPUT]]", object, (host-sycl)
// CHK-UBACTIONS: 2: clang-offload-unbundler, {1}, object, (host-sycl)
// CHK-UBACTIONS: 3: linker, {0, 2}, image, (host-sycl)
// CHK-UBACTIONS: 4: spirv-to-ir-wrapper, {2}, ir, (device-sycl)
// CHK-UBACTIONS: 5: linker, {4}, ir, (device-sycl)
// CHK-UBACTIONS: 6: sycl-post-link, {5}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 7: file-table-tform, {6}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 8: llvm-spirv, {7}, tempfilelist, (device-sycl)
// CHK-UBACTIONS: 9: file-table-tform, {6, 8}, tempfiletable, (device-sycl)
// CHK-UBACTIONS: 10: clang-offload-wrapper, {9}, object, (device-sycl)
// CHK-UBACTIONS: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {10}, image

/// ###########################################################################

/// Check separate compilation with offloading - unbundling with source
// RUN:   touch %t.o
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
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
// CHK-UBUACTIONS: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {5}, "device-sycl (spir64-unknown-unknown)" {8}, c++-cpp-output
// CHK-UBUACTIONS: 10: compiler, {9}, ir, (host-sycl)
// CHK-UBUACTIONS: 11: backend, {10}, assembler, (host-sycl)
// CHK-UBUACTIONS: 12: assembler, {11}, object, (host-sycl)
// CHK-UBUACTIONS: 13: linker, {0, 2, 12}, image, (host-sycl)
// CHK-UBUACTIONS: 14: spirv-to-ir-wrapper, {2}, ir, (device-sycl)
// CHK-UBUACTIONS: 15: linker, {14, 8}, ir, (device-sycl)
// CHK-UBUACTIONS: 16: sycl-post-link, {15}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 17: file-table-tform, {16}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 18: llvm-spirv, {17}, tempfilelist, (device-sycl)
// CHK-UBUACTIONS: 19: file-table-tform, {16, 18}, tempfiletable, (device-sycl)
// CHK-UBUACTIONS: 20: clang-offload-wrapper, {19}, object, (device-sycl)
// CHK-UBUACTIONS: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {13}, "device-sycl (spir64-unknown-unknown)" {20}, image

/// ###########################################################################

/// Check -fsycl-is-device is passed when compiling for the device.
/// also check for SPIR-V binary creation
// RUN:   %clang -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-is-device and emitting to .spv when compiling for the device
/// when using -fsycl-device-obj=spirv
// RUN:   %clang -### -fsycl-device-obj=spirv -fsycl -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s
// RUN:   %clang_cl -### -fsycl-device-obj=spirv -fsycl -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s

// CHK-FSYCL-IS-DEVICE-NO-BITCODE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-link behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// CHK-LINK-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-UB: 2: linker, {1}, image, (device-sycl)
// CHK-LINK-UB: 3: sycl-post-link, {2}, ir, (device-sycl)
// CHK-LINK-UB: 4: file-table-tform, {3}, tempfilelist, (device-sycl)
// CHK-LINK-UB: 5: llvm-spirv, {4}, tempfilelist, (device-sycl)
// CHK-LINK-UB: 6: file-table-tform, {3, 5}, tempfiletable, (device-sycl)
// CHK-LINK-UB: 7: clang-offload-wrapper, {6}, object, (device-sycl)
// CHK-LINK-UB: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, object

/// ###########################################################################

/// Check -fsycl-link behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-LINK: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-LINK: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK: 3: linker, {2}, image, (device-sycl)
// CHK-LINK: 4: sycl-post-link, {3}, ir, (device-sycl)
// CHK-LINK: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-LINK: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-LINK: 7: file-table-tform, {4, 6}, tempfiletable, (device-sycl)
// CHK-LINK: 8: clang-offload-wrapper, {7}, object, (device-sycl)
// CHK-LINK: 9: offload, "device-sycl (spir64-unknown-unknown)" {8}, object

/// ###########################################################################

/// Check for default linking of -lsycl with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-SYCL %s
// CHECK-LD-SYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SYCL: "-lsycl"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl -nolibsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOLIBSYCL %s
// CHECK-LD-NOLIBSYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOLIBSYCL-NOT: "-lsycl"

/// Check no SYCL runtime is linked with -nostdlib
// RUN: %clang -fsycl -nostdlib -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOSTDLIB %s
// CHECK-LD-NOSTDLIB: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOSTDLIB-NOT: "-lsycl"

/// Check for default linking of syclN.lib with -fsycl usage
// RUN: %clang -fsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// RUN: %clang_cl -fsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-CL %s
// CHECK-LINK-SYCL-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-SYCL-CL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"
// CHECK-LINK-SYCL: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl -nolibsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL %s
// RUN: %clang_cl -fsycl -nolibsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL-CL %s
// CHECK-LINK-NOLIBSYCL-CL-NOT: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOLIBSYCL: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOLIBSYCL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check SYCL runtime is linked despite -nostdlib on Windows, this is
/// necessary for the Windows Clang CMake to work
// RUN: %clang -fsycl -nostdlib -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB %s
// RUN: %clang_cl -fsycl -nostdlib %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB-CL %s
// CHECK-LINK-NOSTDLIB-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOSTDLIB: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOSTDLIB: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check sycld.lib is chosen with /MDd
// RUN:  %clang_cl -fsycl /MDd %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
/// Check sycld is pulled in when msvcrtd is used
// RUN: %clangxx -fsycl -Xclang --dependent-lib=msvcrtd \
// RUN:   -target x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
// CHECK-LINK-SYCL-DEBUG: "--dependent-lib=sycl{{[0-9]*}}d"
// CHECK-LINK-SYCL-DEBUG-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check "-spirv-allow-unknown-intrinsics=llvm.genx." option is emitted for llvm-spirv tool
// RUN: %clangxx %s -fsycl -### 2>&1 | FileCheck %s --check-prefix=CHK-ALLOW-INTRIN
// CHK-ALLOW-INTRIN: llvm-spirv{{.*}}-spirv-allow-unknown-intrinsics=llvm.genx.

/// ###########################################################################

/// Check -Xsycl-target-frontend does not trigger an error when no -fsycl-targets is specified
// RUN:   %clang -### -fsycl -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-TARGET-ERROR %s
// CHK-NO-FSYCL-TARGET-ERROR-NOT: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS %s
// CHK-TOOLS-OPTS: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-DFOO1 -DFOO2"

/// Check for implied options (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: clang-offload-wrapper{{.*}} "-compile-opts=-g{{.*}}-DFOO1 -DFOO2"

/// Check for implied options (-O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang_cl -### -fsycl -fsycl-targets=spir64-unknown-unknown -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -O0 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O2 %s
// CHK-TOOLS-IMPLIED-OPTS-O0-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-cl-opt-disable"
// CHK-TOOLS-IMPLIED-OPTS-O2-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-cl-opt-disable"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64-unknown-unknown -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS2 %s
// CHK-TOOLS-OPTS2: clang-offload-wrapper{{.*}} "-link-opts=-DFOO1 -DFOO2"

/// -fsycl-disable-range-rounding settings
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:        -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// CHK-DISABLE-RANGE-ROUNDING: "-fsycl-disable-range-rounding"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:        -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:        -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// CHK-RANGE-ROUNDING-NOT: "-fsycl-disable-range-rounding"

/// ###########################################################################

/// Verify that triple-boundarch pairs are correct with multi-targetting
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=nvptx64-nvidia-cuda,spir64 -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 10: linker, {9}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 11: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 12: preprocessor, {11}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 13: compiler, {12}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 14: linker, {13}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 15: sycl-post-link, {14}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 16: file-table-tform, {15}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 17: backend, {16}, assembler, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 18: assembler, {17}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 19: linker, {17, 18}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 20: foreach, {16, 19}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 21: file-table-tform, {15, 20}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 22: clang-offload-wrapper, {21}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 23: linker, {5}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 24: sycl-post-link, {23}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 25: file-table-tform, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 26: llvm-spirv, {25}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 27: file-table-tform, {24, 26}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 29: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {22}, "device-sycl (spir64-unknown-unknown)" {28}, image

/// Check the behaviour however with swapped -fsycl-targets
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,nvptx64-nvidia-cuda -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 3: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 5: compiler, {4}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {5}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 10: linker, {9}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 11: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 13: compiler, {12}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 14: linker, {13}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 20: linker, {5}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 21: sycl-post-link, {20}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 22: file-table-tform, {21}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 23: backend, {22}, assembler, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 24: assembler, {23}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 25: linker, {23, 24}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 26: foreach, {22, 25}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 27: file-table-tform, {21, 26}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 28: clang-offload-wrapper, {27}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 29: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown)" {19}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {28}, image

/// ###########################################################################

// Check if valid bound arch behaviour occurs when compiling for spir-v,nvidia-gpu, and amd-gpu
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,nvptx-nvidia-cuda,amdgcn-amd-amdhsa -Xsycl-target-backend=nvptx-nvidia-cuda --offload-arch=sm_75 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx908 -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD %s
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 3: input, "[[INPUT]]", c++, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 5: compiler, {4}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (amdgcn-amd-amdhsa:gfx908)" {5}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 10: linker, {9}, image, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 11: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 12: preprocessor, {11}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 13: compiler, {12}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 14: linker, {13}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 15: sycl-post-link, {14}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 16: file-table-tform, {15}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 17: llvm-spirv, {16}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 18: file-table-tform, {15, 17}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 20: input, "[[INPUT]]", c++, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 21: preprocessor, {20}, c++-cpp-output, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 22: compiler, {21}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 23: linker, {22}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 24: sycl-post-link, {23}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 25: file-table-tform, {24}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 26: backend, {25}, assembler, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 27: assembler, {26}, object, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 28: linker, {26, 27}, cuda-fatbin, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 29: foreach, {25, 28}, cuda-fatbin, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 30: file-table-tform, {24, 29}, tempfiletable, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 31: clang-offload-wrapper, {30}, object, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 32: linker, {5}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 33: sycl-post-link, {32}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 34: file-table-tform, {33}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 35: backend, {34}, assembler, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 36: assembler, {35}, object, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 37: linker, {36}, image, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 38: linker, {37}, hip-fatbin, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 39: foreach, {34, 38}, hip-fatbin, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 40: file-table-tform, {33, 39}, tempfiletable, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 41: clang-offload-wrapper, {40}, object, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 42: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown)" {19}, "device-sycl (nvptx-nvidia-cuda:sm_75)" {31}, "device-sycl (amdgcn-amd-amdhsa:gfx908)" {41}, image

/// -fsycl with /Fo testing
// RUN: %clang_cl -fsycl /Fosomefile.obj -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=FO-CHECK %s
// FO-CHECK: clang{{.*}} "-fsycl-int-header=[[HEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// FO-CHECK: clang{{.*}} "-include" "[[HEADER]]" {{.*}} "-o" "[[OUTPUT2:.+\.obj]]"
// FO-CHECK: clang-offload-bundler{{.*}} "-output=somefile.obj" "-input=[[OUTPUT1]]" "-input=[[OUTPUT2]]"

/// passing of a library should not trigger the unbundler
// RUN: touch %t.a
// RUN: touch %t.lib
// RUN: %clang -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.a %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// RUN: %clang_cl -ccc-print-phases -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.lib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// LIB-UNBUNDLE-CHECK-NOT: clang-offload-unbundler

/// passing of only a library should not create a device link
// RUN: %clang -ccc-print-phases -fsycl -lsomelib 2>&1 \
// RUN:  | FileCheck -check-prefix=LIB-NODEVICE %s
// LIB-NODEVICE: 0: input, "somelib", object, (host-sycl)
// LIB-NODEVICE: 1: linker, {0}, image, (host-sycl)
// LIB-NODEVICE-NOT: linker, {{.*}}, spirv, (device-sycl)

// Checking for an error if c-compilation is forced
// RUN: not %clangxx -### -c -fsycl -xc %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: not %clangxx -### -c -fsycl -xc-header %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: not %clangxx -### -c -fsycl -xcpp-output %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
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
// CHK-POST-LINK-OPT-LEVEL-O0: sycl-post-link{{.*}} "-O2"
// CHK-POST-LINK-OPT-LEVEL-O1: sycl-post-link{{.*}} "-O1"
// CHK-POST-LINK-OPT-LEVEL-O2: sycl-post-link{{.*}} "-O2"
// CHK-POST-LINK-OPT-LEVEL-O3: sycl-post-link{{.*}} "-O3"
// CHK-POST-LINK-OPT-LEVEL-Os: sycl-post-link{{.*}} "-Os"
// CHK-POST-LINK-OPT-LEVEL-Oz: sycl-post-link{{.*}} "-Oz"

// Verify header search dirs are added with -fsycl
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// RUN: %clang_cl -### -fsycl %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-device"
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT:[^"]*]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include"
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-host"
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include"

/// Check for option incompatibility with -fsycl
// RUN:   not %clang -### -fsycl -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-ffreestanding
// RUN:   not %clang -### -fsycl -static-libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-static-libstdc++
// CHK-INCOMPATIBILITY: error: '[[INCOMPATOPT]]' is not supported with '-fsycl'

/// Using -fsyntax-only with -fsycl should not emit IR
// RUN:   %clang -### -fsycl -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYNTAX-ONLY %s
// RUN:   %clang -### -fsycl -fsycl-device-only -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYNTAX-ONLY %s
// CHK-FSYNTAX-ONLY-NOT: "-emit-llvm-bc"
// CHK-FSYNTAX-ONLY: "-fsyntax-only"

// Emit warning for treating 'c' input as 'c++' when -fsycl is used
// RUN: %clang -### -fsycl  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang_cl -### -fsycl  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// FSYCL-CHECK: warning: treating 'c' input as 'c++' when -fsycl is used [-Wexpected-file-type]

/// Check for linked sycl lib when using -fpreview-breaking-changes with -fsycl
// RUN: %clang -### -fsycl -fpreview-breaking-changes -target x86_64-unknown-windows-msvc %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-CHECK %s
// RUN: %clang_cl -### -fsycl -fpreview-breaking-changes %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-CL %s
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK: -defaultlib:sycl{{[0-9]*}}-preview.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-CL: "--dependent-lib=sycl{{[0-9]*}}-preview"

/// Check for linked sycl lib when using -fpreview-breaking-changes with -fsycl
// RUN: %clang -### -fsycl -fpreview-breaking-changes -target x86_64-unknown-windows-msvc -Xclang --dependent-lib=msvcrtd %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK %s
// RUN: %clang_cl -### -fsycl -fpreview-breaking-changes /MDd %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK %s
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK: --dependent-lib=sycl{{[0-9]*}}-previewd
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}-preview.lib
