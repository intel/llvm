///
/// Perform several driver tests for SYCL offloading
///

// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl --no-offload-new-driver -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// RUN:   not %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: SYCL target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl --no-offload-new-driver -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// RUN:   not %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// CHK-INVALID-REAL-TARGET: error: SYCL target is invalid: 'x86_64'

/// ###########################################################################

/// Check warning for empty -fsycl-targets
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// CHK-EMPTY-SYCLTARGETS: warning: joined argument expects additional value: '-fsycl-targets='

/// ###########################################################################

/// Check error for no -fsycl --no-offload-new-driver option
// RUN:   not %clang -### -fsycl-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// RUN:   not %clang_cl -### -fsycl-targets=spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-targets' must be used in conjunction with '-fsycl' to enable offloading
// RUN: not %clang -### -fsycl-link  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-LINK %s
// CHK-NO-FSYCL-LINK: error: '-fsycl-link' must be used in conjunction with '-fsycl' to enable offloading

/// ###########################################################################

/// Validate SYCL option values
// RUN:   not %clang -### -fsycl-device-code-split=bad_value -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-device-code-split %s
// RUN:   not %clang -### -fsycl-link=bad_value -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-link %s
// CHK-SYCL-BAD-OPT-VALUE: error: invalid argument 'bad_value' to [[option]]=

/// Check no error for -fsycl-targets with good triple
// RUN:   %clang -### -fsycl-targets=spir-unknown-unknown -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64 -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang_cl -### -fsycl-targets=spir-unknown-unknown -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv32-unknown-unknown -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv64-unknown-unknown -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv32 -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv64 -fsycl --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// CHK-SYCL-TARGET-NOT: error: SYCL target is invalid

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: SYCL offloading target 'spir64-unknown-unknown' is similar to target 'spir64-unknown-unknown' already specified; will be ignored

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when multiple triples are used.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when an option requiring arguments is passed to it.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// CHK-FSYCL-COMPILER-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target-frontend argument: '-Xsycl-target-frontend -Xsycl-target-frontend', options requiring arguments are unsupported

/// ###########################################################################

/// Check -Xsycl-target-frontend= accepts triple aliases
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=spir64 -Xsycl-target-frontend=spir64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH1=spir64 -check-prefixes=CHK-UNUSED-ARG-WARNING-1,CHK-TARGET-1 %s
// CHK-UNUSED-ARG-WARNING-1-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET-1: clang{{.*}} "-cc1" "-triple" "[[ARCH1]]-unknown-unknown"{{.*}} "-D" "FOO"
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-targets=spirv64 -Xsycl-target-frontend=spirv64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH2=spirv64 -check-prefixes=CHK-UNUSED-ARG-WARNING-2,CHK-TARGET-2 %s
// CHK-UNUSED-ARG-WARNING-2-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET-2: clang{{.*}} "-cc1" "-triple" "[[ARCH2]]-unknown-unknown"{{.*}} "-D" "FOO"

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-device-obj=llvmir
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-device-obj=spirv -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fsycl-device-obj=spirv -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-device-obj=llvmir -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fsycl-device-obj=llvmir -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES %s
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

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.  The same graph should be generated when no -fsycl-targets is used
/// The same phase graph will be used with -fsycl-device-obj=llvmir
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spirv64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-2,CHK-PHASES-DEFAULT-MODE-2 %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fsycl-targets=spirv64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-2,CHK-PHASES-CL-MODE-2 %s
// CHK-PHASES-2: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-2: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-2: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-2: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES-2: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-DEFAULT-MODE-2: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spirv64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-CL-MODE-2: 5: offload, "host-sycl (x86_64-pc-windows-msvc)" {1}, "device-sycl (spirv64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASES-2: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-2: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-2: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-2: 9: linker, {4}, ir, (device-sycl)
// CHK-PHASES-2: 10: sycl-post-link, {9}, tempfiletable, (device-sycl)
// CHK-PHASES-2: 11: file-table-tform, {10}, tempfilelist, (device-sycl)
// CHK-PHASES-2: 12: llvm-spirv, {11}, tempfilelist, (device-sycl)
// CHK-PHASES-2: 13: file-table-tform, {10, 12}, tempfiletable, (device-sycl)
// CHK-PHASES-2: 14: clang-offload-wrapper, {13}, object, (device-sycl)
// CHK-PHASES-2: 15: offload, "device-sycl (spirv64-unknown-unknown)" {14}, object
// CHK-PHASES-2: 16: linker, {8, 15}, image, (host-sycl)

/// ###########################################################################

/// Check the compilation flow to verify that the integrated header is filtered
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -c %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHK-INT-HEADER
// CHK-INT-HEADER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// CHK-INT-HEADER: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include-internal-header" "[[INPUT1]]" "-dependency-filter" "[[INPUT1]]" {{.*}} "-o" "[[OUTPUT2:.+\.o]]"
// CHK-INT-HEADER: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown,host-x86_64-unknown-linux-gnu" {{.*}} "-input=[[OUTPUT1]]" "-input=[[OUTPUT2]]"

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
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

/// Compilation check with -lstdc++ (treated differently than regular lib)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -lstdc++ -fsycl --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LIB-STDCXX %s
// CHK-LIB-STDCXX: ld{{.*}} "-lstdc++"
// CHK-LIB-STDCXX-NOT: clang-offload-bundler{{.*}}
// CHK-LIB-STDCXX-NOT: llvm-link{{.*}} "-lstdc++"

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s %t.c 2>&1 \
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

/// Check separate compilation with offloading - bundling actions
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -c -o %t.o -lsomelib -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-BUACTIONS %s
// CHK-BUACTIONS: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-BUACTIONS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-BUACTIONS: 2: compiler, {1}, ir, (device-sycl)
// CHK-BUACTIONS: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// CHK-BUACTIONS: 4: input, "[[INPUT]]", c++, (host-sycl)
// CHK-BUACTIONS: 5: preprocessor, {4}, c++-cpp-output, (host-sycl)
// CHK-BUACTIONS: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {5}, "device-sycl (spir64-unknown-unknown)" {2}, c++-cpp-output
// CHK-BUACTIONS: 7: compiler, {6}, ir, (host-sycl)
// CHK-BUACTIONS: 8: backend, {7}, assembler, (host-sycl)
// CHK-BUACTIONS: 9: assembler, {8}, object, (host-sycl)
// CHK-BUACTIONS: 10: clang-offload-bundler, {3, 9}, object, (host-sycl)

/// ###########################################################################

/// Check separate compilation with offloading - unbundling actions
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64 %t.o 2>&1 \
// RUN:   | FileCheck -DINPUT=%t.o -check-prefix=CHK-UBACTIONS %s
// RUN:   mkdir -p %t_dir
// RUN:   touch %t_dir/dummy
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -o %t.out -lsomelib -fsycl-targets=spir64-unknown-unknown %t_dir/dummy 2>&1 \
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
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
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

/// Check -fsycl-is-device is passed when compiling for the device.
/// also check for SPIR-V binary creation
// RUN:   %clang -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-is-device and emitting to .spv when compiling for the device
/// when using -fsycl-device-obj=spirv
// RUN:   %clang -### -fsycl-device-obj=spirv -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s
// RUN:   %clang_cl -### -fsycl-device-obj=spirv -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s

// CHK-FSYCL-IS-DEVICE-NO-BITCODE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-link behaviors unbundle
// RUN:   touch %t.o
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-UB %s
// CHK-LINK-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-UB: 2: spirv-to-ir-wrapper, {1}, ir, (device-sycl)
// CHK-LINK-UB: 3: linker, {2}, ir, (device-sycl)
// CHK-LINK-UB: 4: sycl-post-link, {3}, tempfiletable, (device-sycl)
// CHK-LINK-UB: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-LINK-UB: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-LINK-UB: 7: file-table-tform, {4, 6}, tempfiletable, (device-sycl)
// CHK-LINK-UB: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, tempfiletable
// CHK-LINK-UB: 9: clang-offload-wrapper, {8}, ir, (host-sycl)
// CHK-LINK-UB: 10: backend, {9}, assembler, (host-sycl)
// CHK-LINK-UB: 11: assembler, {10}, object, (host-sycl)

/// Check -fsycl-link tool calls
// RUN:   %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -o %t.out \
// RUN:            -fsycl-targets=spir64_gen -fsycl-link \
// RUN:            -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYCL-LINK-UB,CHK-FSYCL-LINK-UB-LIN %s
// RUN:   %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -o %t.out \
// RUN:            -fsycl-targets=spir64_gen -fsycl-link \
// RUN:            -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYCL-LINK-UB,CHK-FSYCL-LINK-UB-WIN %s
// CHK-FSYCL-LINK-UB: clang-offload-bundler{{.*}} "-type=o" "-targets=host{{.*}},sycl-spir64_gen-unknown-unknown" "-input=[[INPUT:.+\.o]]" "-output={{.*}}" "-output=[[DEVICE_O:.+]]" "-unbundle"
// CHK-FSYCL-LINK-UB: spirv-to-ir-wrapper{{.*}} "[[DEVICE_O]]" "-o" "[[DEVICE_BC:.+\.bc]]"
// CHK-FSYCL-LINK-UB: llvm-link{{.*}} "[[DEVICE_BC]]"
// CHK-FSYCL-LINK-UB: sycl-post-link{{.*}} "-o" "[[POST_LINK_TABLE:.+\.table]]"
// CHK-FSYCL-LINK-UB: file-table-tform{{.*}} "-o" "[[TFORM_TABLE:.+\.txt]]" "[[POST_LINK_TABLE]]"
// CHK-FSYCL-LINK-UB: llvm-spirv{{.*}} "-o" "[[SPIRV:.+\.txt]]"{{.*}} "[[TFORM_TABLE]]"
// CHK-FSYCL-LINK-UB-LIN: ocloc{{.*}} "-output" "[[OCLOC_OUT:.+\.out]]"
// CHK-FSYCL-LINK-UB-WIN: ocloc{{.*}} "-output" "[[OCLOC_OUT:.+\.exe]]"
// CHK-FSYCL-LINK-UB: file-table-tform{{.*}} "-o" "[[TFORM_TABLE2:.+\.table]]" "[[POST_LINK_TABLE]]" "[[OCLOC_OUT]]"
// CHK-FSYCL-LINK-UB: clang-offload-wrapper{{.*}} "-o" "[[WRAPPER_OUT:.+\.bc]]"{{.*}} "-batch" "[[TFORM_TABLE2]]"
// CHK-FSYCL-LINK-UB: clang{{.*}} "-cc1"{{.*}} "-o" "{{.*}}.out" "-x" "ir" "[[WRAPPER_OUT]]"

/// Check -fsycl-link AOT unbundle
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu \
// RUN:     -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code \
// RUN:     -fsycl-targets=spir64_gen -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-AOT-UB %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc \
// RUN:     -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code \
// RUN:     -fsycl-targets=spir64_gen -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-AOT-UB %s
// CHK-LINK-AOT-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-AOT-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-AOT-UB: 2: spirv-to-ir-wrapper, {1}, ir, (device-sycl)
// CHK-LINK-AOT-UB: 3: linker, {2}, ir, (device-sycl)
// CHK-LINK-AOT-UB: 4: sycl-post-link, {3}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-UB: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-UB: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-UB: 7: backend-compiler, {6}, image, (device-sycl)
// CHK-LINK-AOT-UB: 8: file-table-tform, {4, 7}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-UB: 9: offload, "device-sycl (spir64_gen-unknown-unknown)" {8}, tempfiletable
// CHK-LINK-AOT-UB: 10: clang-offload-wrapper, {9}, ir, (host-sycl)
// CHK-LINK-AOT-UB: 11: backend, {10}, assembler, (host-sycl)
// CHK-LINK-AOT-UB: 12: assembler, {11}, object, (host-sycl)

/// Check -fsycl-link AOT/JIT unbundle
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu \
// RUN:     -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code \
// RUN:     -fsycl-targets=spir64_gen,spir64 \
// RUN:     -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-AOT-JIT-UB %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc \
// RUN:     -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code \
// RUN:     -fsycl-targets=spir64_gen,spir64 \
// RUN:     -fno-sycl-device-lib=all %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK-AOT-JIT-UB %s
// CHK-LINK-AOT-JIT-UB: 0: input, "[[INPUT:.+\.o]]", object
// CHK-LINK-AOT-JIT-UB: 1: clang-offload-unbundler, {0}, object
// CHK-LINK-AOT-JIT-UB: 2: spirv-to-ir-wrapper, {1}, ir, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 3: linker, {2}, ir, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 4: sycl-post-link, {3}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 7: backend-compiler, {6}, image, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 8: file-table-tform, {4, 7}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 9: offload, "device-sycl (spir64_gen-unknown-unknown)" {8}, tempfiletable
// CHK-LINK-AOT-JIT-UB: 10: spirv-to-ir-wrapper, {1}, ir, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 11: linker, {10}, ir, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 15: file-table-tform, {12, 14}, tempfiletable, (device-sycl)
// CHK-LINK-AOT-JIT-UB: 16: offload, "device-sycl (spir64-unknown-unknown)" {15}, tempfiletable
// CHK-LINK-AOT-JIT-UB: 17: clang-offload-wrapper, {9, 16}, ir, (host-sycl)
// CHK-LINK-AOT-JIT-UB: 18: backend, {17}, assembler, (host-sycl)
// CHK-LINK-AOT-JIT-UB: 19: assembler, {18}, object, (host-sycl)

/// ###########################################################################

/// Check -fsycl-link behaviors from source
// RUN:   %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// RUN:   %clang_cl -### -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -o %t.out -fsycl-link -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-LINK %s
// CHK-LINK: 0: input, "[[INPUT:.+\.c]]", c++, (device-sycl)
// CHK-LINK: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-LINK: 2: compiler, {1}, ir, (device-sycl)
// CHK-LINK: 3: linker, {2}, ir, (device-sycl)
// CHK-LINK: 4: sycl-post-link, {3}, tempfiletable, (device-sycl)
// CHK-LINK: 5: file-table-tform, {4}, tempfilelist, (device-sycl)
// CHK-LINK: 6: llvm-spirv, {5}, tempfilelist, (device-sycl)
// CHK-LINK: 7: file-table-tform, {4, 6}, tempfiletable, (device-sycl)
// CHK-LINK: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, tempfiletable
// CHK-LINK: 9: clang-offload-wrapper, {8}, ir, (host-sycl)
// CHK-LINK: 10: backend, {9}, assembler, (host-sycl)
// CHK-LINK: 11: assembler, {10}, object, (host-sycl)

/// ###########################################################################

/// Check for default linking of -lsycl with -fsycl --no-offload-new-driver usage
// RUN: %clang -fsycl --no-offload-new-driver -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-SYCL %s
// CHECK-LD-SYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SYCL: "-lsycl"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl --no-offload-new-driver -nolibsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOLIBSYCL %s
// CHECK-LD-NOLIBSYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOLIBSYCL-NOT: "-lsycl"

/// Check no SYCL runtime is linked with -nostdlib
// RUN: %clang -fsycl --no-offload-new-driver -nostdlib -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOSTDLIB %s
// CHECK-LD-NOSTDLIB: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOSTDLIB-NOT: "-lsycl"

/// Check for default linking of syclN.lib with -fsycl --no-offload-new-driver usage
// RUN: %clang -fsycl --no-offload-new-driver -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// RUN: %clang_cl -fsycl --no-offload-new-driver %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-CL %s
// CHECK-LINK-SYCL-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-SYCL-CL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"
// CHECK-LINK-SYCL: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl --no-offload-new-driver -nolibsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL %s
// RUN: %clang_cl -fsycl --no-offload-new-driver -nolibsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL-CL %s
// CHECK-LINK-NOLIBSYCL-CL-NOT: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOLIBSYCL: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOLIBSYCL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check SYCL runtime is linked despite -nostdlib on Windows, this is
/// necessary for the Windows Clang CMake to work
// RUN: %clang -fsycl --no-offload-new-driver -nostdlib -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB %s
// RUN: %clang_cl -fsycl --no-offload-new-driver -nostdlib %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB-CL %s
// CHECK-LINK-NOSTDLIB-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOSTDLIB: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOSTDLIB: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check sycld.lib is chosen with /MDd
// RUN:  %clang_cl -fsycl --no-offload-new-driver /MDd %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
/// Check sycld is pulled in when msvcrtd is used
// RUN: %clangxx -fsycl --no-offload-new-driver -Xclang --dependent-lib=msvcrtd \
// RUN:   -target x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
/// Check sycld is pulled in when -fms-runtime-lib=dll_dbg
// RUN: %clangxx -fsycl --no-offload-new-driver -fms-runtime-lib=dll_dbg \
// RUN:   -target x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
// CHECK-LINK-SYCL-DEBUG: "--dependent-lib=sycl{{[0-9]*}}d"
// CHECK-LINK-SYCL-DEBUG-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check "-spirv-allow-unknown-intrinsics=llvm.genx." option is emitted for llvm-spirv tool
// RUN: %clangxx %s -fsycl --no-offload-new-driver -### 2>&1 | FileCheck %s --check-prefix=CHK-ALLOW-INTRIN
// CHK-ALLOW-INTRIN: llvm-spirv{{.*}}-spirv-allow-unknown-intrinsics=llvm.genx.

/// ###########################################################################

/// Check -Xsycl-target-frontend does not trigger an error when no -fsycl-targets is specified
// RUN:   %clang -### -fsycl --no-offload-new-driver -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-TARGET-ERROR %s
// CHK-NO-FSYCL-TARGET-ERROR-NOT: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS %s
// CHK-TOOLS-OPTS: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-DFOO1 -DFOO2"

/// Check for implied options (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: clang-offload-wrapper{{.*}} "-compile-opts=-g{{.*}}-DFOO1 -DFOO2"

/// Check for implied options (-O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64 -O0 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O2 %s
// CHK-TOOLS-IMPLIED-OPTS-O0-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-cl-opt-disable"
// CHK-TOOLS-IMPLIED-OPTS-O2-NOT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-cl-opt-disable"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS2 %s
// CHK-TOOLS-OPTS2: clang-offload-wrapper{{.*}} "-link-opts=-DFOO1 -DFOO2"

/// -fsycl-range-rounding settings
///
/// // Check that driver flag is passed to cc1
// RUN: %clang -### -fsycl --no-offload-new-driver -fsycl-range-rounding=disable %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-DISABLE %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-FORCE %s
// RUN: %clang -### -fsycl --no-offload-new-driver -fsycl-range-rounding=on %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-ON %s
// CHK-DRIVER-RANGE-ROUNDING-DISABLE: "-cc1{{.*}}-fsycl-range-rounding=disable"
// CHK-DRIVER-RANGE-ROUNDING-FORCE: "-cc1{{.*}}-fsycl-range-rounding=force"
// CHK-DRIVER-RANGE-ROUNDING-ON: "-cc1{{.*}}-fsycl-range-rounding=on"
///
///
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:        -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=spir64 -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:        -O0 -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OVERRIDE-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -Od %s 2>&1 -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OVERRIDE-RANGE-ROUNDING %s
// CHK-DISABLE-RANGE-ROUNDING: "-fsycl-range-rounding=disable"
// CHK-OVERRIDE-RANGE-ROUNDING: "-fsycl-range-rounding=force"
// CHK-OVERRIDE-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=disable"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:        -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:        -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// CHK-RANGE-ROUNDING-NOT: "-fsycl-disable-range-rounding"
// CHK-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=disable"
// CHK-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=force"

/// ###########################################################################

/// Verify that triple-boundarch pairs are correct with multi-targetting
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=nvptx64-nvidia-cuda,spir64 -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 9: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 10: preprocessor, {9}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 11: compiler, {10}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 12: linker, {11}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 13: sycl-post-link, {12}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 14: file-table-tform, {13}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 15: backend, {14}, assembler, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 16: assembler, {15}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 17: linker, {15, 16}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 18: foreach, {14, 17}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 19: file-table-tform, {13, 18}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 20: clang-offload-wrapper, {19}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 21: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {20}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 22: linker, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 26: file-table-tform, {23, 25}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 28: offload, "device-sycl (spir64-unknown-unknown)" {27}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 29: linker, {8, 21, 28}, image, (host-sycl)

// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl \
// RUN:     -fno-sycl-instrument-device-code -fno-sycl-device-lib=all \
// RUN:     -fsycl-targets=nvptx64-nvidia-cuda,spir64_gen \
// RUN:     -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:     -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH2 %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 4: compiler, {3}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64_gen-unknown-unknown)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 9: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 10: preprocessor, {9}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 11: compiler, {10}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 12: linker, {11}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 13: sycl-post-link, {12}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 14: file-table-tform, {13}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 15: backend, {14}, assembler, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 16: assembler, {15}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 17: linker, {15, 16}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 18: foreach, {14, 17}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 19: file-table-tform, {13, 18}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 20: clang-offload-wrapper, {19}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 21: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {20}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 22: linker, {4}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 23: sycl-post-link, {22}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 24: file-table-tform, {23}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 25: llvm-spirv, {24}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 26: backend-compiler, {25}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 27: file-table-tform, {23, 26}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 28: clang-offload-wrapper, {27}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 29: offload, "device-sycl (spir64_gen-unknown-unknown)" {28}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 30: linker, {8, 21, 29}, image, (host-sycl)

/// Check the behaviour however with swapped -fsycl-targets
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,nvptx64-nvidia-cuda -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 2: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 4: compiler, {3}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 9: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 11: compiler, {10}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 12: linker, {11}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 18: offload, "device-sycl (spir64-unknown-unknown)" {17}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 19: linker, {4}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 20: sycl-post-link, {19}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 21: file-table-tform, {20}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 22: backend, {21}, assembler, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 23: assembler, {22}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 24: linker, {22, 23}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 25: foreach, {21, 24}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 26: file-table-tform, {20, 25}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 27: clang-offload-wrapper, {26}, object, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 28: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {27}, object
// CHK-PHASE-MULTI-TARG-BOUND-ARCH-FLIPPED: 29: linker, {8, 18, 28}, image, (host-sycl)

/// ###########################################################################

// Check if valid bound arch behaviour occurs when compiling for spir-v,nvidia-gpu, and amd-gpu
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64,nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_75 -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx908 -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD %s
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 2: input, "[[INPUT]]", c++, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 4: compiler, {3}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (amdgcn-amd-amdhsa:gfx908)" {4}, c++-cpp-output
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 9: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 11: compiler, {10}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 12: linker, {11}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 13: sycl-post-link, {12}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 14: file-table-tform, {13}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 15: llvm-spirv, {14}, tempfilelist, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 16: file-table-tform, {13, 15}, tempfiletable, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 18: offload, "device-sycl (spir64-unknown-unknown)" {17}, object
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 19: input, "[[INPUT]]", c++, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 20: preprocessor, {19}, c++-cpp-output, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 21: compiler, {20}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 22: linker, {21}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 23: sycl-post-link, {22}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 24: file-table-tform, {23}, ir, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 25: backend, {24}, assembler, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 26: assembler, {25}, object, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 27: linker, {25, 26}, cuda-fatbin, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 28: foreach, {24, 27}, cuda-fatbin, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 29: file-table-tform, {23, 28}, tempfiletable, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 30: clang-offload-wrapper, {29}, object, (device-sycl, sm_75)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 31: offload, "device-sycl (nvptx64-nvidia-cuda:sm_75)" {30}, object
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 32: linker, {4}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 33: sycl-post-link, {32}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 34: file-table-tform, {33}, ir, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 35: backend, {34}, assembler, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 36: assembler, {35}, object, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 37: linker, {36}, image, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 38: linker, {37}, hip-fatbin, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 39: foreach, {34, 38}, hip-fatbin, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 40: file-table-tform, {33, 39}, tempfiletable, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 41: clang-offload-wrapper, {40}, object, (device-sycl, gfx908)
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 42: offload, "device-sycl (amdgcn-amd-amdhsa:gfx908)" {41}, object
// CHK-PHASE-MULTI-TARG-SPIRV-NVIDIA-AMD: 43: linker, {8, 18, 31, 42}, image, (host-sycl)

/// -fsycl --no-offload-new-driver with /Fo testing
// RUN: %clang_cl -fsycl --no-offload-new-driver /Fosomefile.obj -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=FO-CHECK %s
// FO-CHECK: clang{{.*}} "-fsycl-int-header=[[HEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[OUTPUT1:.+\.bc]]"
// FO-CHECK: clang{{.*}} "-include-internal-header" "[[HEADER]]" {{.*}} "-o" "[[OUTPUT2:.+\.obj]]"
// FO-CHECK: clang-offload-bundler{{.*}} "-output=somefile.obj" "-input=[[OUTPUT1]]" "-input=[[OUTPUT2]]"

/// passing of a library should not trigger the unbundler
// RUN: touch %t.a
// RUN: touch %t.lib
// RUN: %clang -ccc-print-phases -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.a %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// RUN: %clang_cl -ccc-print-phases -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t.lib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LIB-UNBUNDLE-CHECK %s
// LIB-UNBUNDLE-CHECK-NOT: clang-offload-unbundler

/// passing of only a library should not create a device link
// RUN: %clang -ccc-print-phases -fsycl --no-offload-new-driver -lsomelib 2>&1 \
// RUN:  | FileCheck -check-prefix=LIB-NODEVICE %s
// LIB-NODEVICE: 0: input, "somelib", object, (host-sycl)
// LIB-NODEVICE: 1: linker, {0}, image, (host-sycl)
// LIB-NODEVICE-NOT: linker, {{.*}}, spirv, (device-sycl)

// Checking for an error if c-compilation is forced
// RUN: not %clangxx -### -c -fsycl --no-offload-new-driver -xc %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: not %clangxx -### -c -fsycl --no-offload-new-driver -xc-header %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// RUN: not %clangxx -### -c -fsycl --no-offload-new-driver -xcpp-output %s 2>&1 | FileCheck -check-prefixes=CHECK_XC_FSYCL %s
// CHECK_XC_FSYCL: '-x c{{.*}}' must not be used in conjunction with '-fsycl'

// -std=c++17 check (check all 3 compilations)
// RUN: %clangxx -### -c -fsycl --no-offload-new-driver -xc++ %s 2>&1 | FileCheck -check-prefix=CHECK-STD %s
// RUN: %clang_cl -### -c -fsycl --no-offload-new-driver -TP %s 2>&1 | FileCheck -check-prefix=CHECK-STD %s
// CHECK-STD: clang{{.*}} "-emit-llvm-bc" {{.*}} "-std=c++17"
// CHECK-STD: clang{{.*}} "-emit-obj" {{.*}} "-std=c++17"

// -std=c++17 override check
// RUN: %clangxx -### -c -fsycl --no-offload-new-driver -std=c++14 -xc++ %s 2>&1 | FileCheck -check-prefix=CHECK-STD-OVR %s
// RUN: %clang_cl -### -c -fsycl --no-offload-new-driver /std:c++14 -TP %s 2>&1 | FileCheck -check-prefix=CHECK-STD-OVR %s
// CHECK-STD-OVR: clang{{.*}} "-emit-llvm-bc" {{.*}} "-std=c++14"
// CHECK-STD-OVR: clang{{.*}} "-emit-obj" {{.*}} "-std=c++14"
// CHECK-STD-OVR-NOT: clang{{.*}} "-std=c++17"

// Check sycl-post-link optimization level.
// Default is O2
// RUN:   %clang    -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// Common options for %clang and %clang_cl
// RUN:   %clang    -### -fsycl --no-offload-new-driver -O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// RUN:   %clang    -### -fsycl --no-offload-new-driver -O2 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O2
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /O2 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// RUN:   %clang    -### -fsycl --no-offload-new-driver -Os %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /Os %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// %clang options
// RUN:   %clang    -### -fsycl --no-offload-new-driver -O0 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O0
// RUN:   %clang    -### -fsycl --no-offload-new-driver -Ofast %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// RUN:   %clang    -### -fsycl --no-offload-new-driver -O3 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// RUN:   %clang    -### -fsycl --no-offload-new-driver -Oz %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Oz
// RUN:   %clang    -### -fsycl --no-offload-new-driver -Og %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// %clang_cl options
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /Od %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O0
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /Ot %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O3
// only the last option is considered
// RUN:   %clang    -### -fsycl --no-offload-new-driver -O2 -O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-O1
// RUN:   %clang_cl -### -fsycl --no-offload-new-driver /O2 /O1 %s 2>&1 | FileCheck %s -check-prefixes=CHK-POST-LINK-OPT-LEVEL-Os
// CHK-POST-LINK-OPT-LEVEL-O0: sycl-post-link{{.*}} "-O2"
// CHK-POST-LINK-OPT-LEVEL-O1: sycl-post-link{{.*}} "-O1"
// CHK-POST-LINK-OPT-LEVEL-O2: sycl-post-link{{.*}} "-O2"
// CHK-POST-LINK-OPT-LEVEL-O3: sycl-post-link{{.*}} "-O3"
// CHK-POST-LINK-OPT-LEVEL-Os: sycl-post-link{{.*}} "-Os"
// CHK-POST-LINK-OPT-LEVEL-Oz: sycl-post-link{{.*}} "-Oz"

// Verify header search dirs are added with -fsycl
// RUN: %clang -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// RUN: %clang_cl -### -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s -check-prefixes=CHECK-HEADER-DIR
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-device"
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT:[^"]*]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include"
// CHECK-HEADER-DIR: clang{{.*}} "-fsycl-is-host"
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// CHECK-HEADER-DIR-NOT: -internal-isystem
// CHECK-HEADER-DIR-SAME: "-internal-isystem" "[[ROOT]]bin{{[/\\]+}}..{{[/\\]+}}include"

/// Check for option incompatibility with -fsycl
// RUN:   not %clang -### -fsycl --no-offload-new-driver -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-ffreestanding
// RUN:   not %clang -### -fsycl --no-offload-new-driver -static-libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-static-libstdc++
// CHK-INCOMPATIBILITY: error: invalid argument '[[INCOMPATOPT]]' not allowed with '-fsycl'

/// Using -fsyntax-only with -fsycl --no-offload-new-driver should not emit IR
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYNTAX-ONLY %s
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-only -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYNTAX-ONLY %s
// CHK-FSYNTAX-ONLY-NOT: "-emit-llvm-bc"
// CHK-FSYNTAX-ONLY: "-fsyntax-only"

// Emit warning for treating 'c' input as 'c++' when -fsycl --no-offload-new-driver is used
// RUN: %clang -### -fsycl --no-offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// FSYCL-CHECK: warning: treating 'c' input as 'c++' when -fsycl is used [-Wexpected-file-type]

/// Check for linked sycl lib when using -fpreview-breaking-changes with -fsycl
// RUN: %clang -### -fsycl --no-offload-new-driver -fpreview-breaking-changes -target x86_64-unknown-windows-msvc %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-CHECK %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fpreview-breaking-changes %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-CL %s
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK: -defaultlib:sycl{{[0-9]*}}-preview.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-CHECK-CL: "--dependent-lib=sycl{{[0-9]*}}-preview"

/// Check for linked sycl lib when using -fpreview-breaking-changes with -fsycl
// RUN: %clang -### -fsycl --no-offload-new-driver -fpreview-breaking-changes -target x86_64-unknown-windows-msvc -Xclang --dependent-lib=msvcrtd %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fpreview-breaking-changes /MDd %s 2>&1 | FileCheck -check-prefix FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK %s
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK: --dependent-lib=sycl{{[0-9]*}}-previewd
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}.lib
// FSYCL-PREVIEW-BREAKING-CHANGES-DEBUG-CHECK-NOT: -defaultlib:sycl{{[0-9]*}}-preview.lib

// Check if fsycl-targets correctly processes multiple NVidia
// and AMD GPU targets.
// RUN:   %clang -### -fsycl -fsycl-targets=nvidia_gpu_sm_60,nvidia_gpu_sm_70 -nocudalib --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-MACRO-SM-60,CHK-MACRO-SM-70 %s
// CHK-MACRO-SM-60: clang{{.*}} "-fsycl-is-device"{{.*}} "-D__SYCL_TARGET_NVIDIA_GPU_SM_60__"{{.*}}
// CHK-MACRO-SM-70: clang{{.*}} "-fsycl-is-device"{{.*}} "-D__SYCL_TARGET_NVIDIA_GPU_SM_70__"{{.*}}
// RUN:   %clang -### -fsycl -fsycl-targets=amd_gpu_gfx90a,amd_gpu_gfx90c -fno-sycl-libspirv -nogpulib --no-offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-MACRO-GFX90A,CHK-MACRO-GFX90C %s
// CHK-MACRO-GFX90A: clang{{.*}} "-fsycl-is-device"{{.*}} "-D__SYCL_TARGET_AMD_GPU_GFX90A__"{{.*}}
// CHK-MACRO-GFX90C: clang{{.*}} "-fsycl-is-device"{{.*}} "-D__SYCL_TARGET_AMD_GPU_GFX90C__"{{.*}}

