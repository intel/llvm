///
/// Perform several driver tests for SYCL offloading
///

// REQUIRES: x86-registered-target

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl --offload-new-driver -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// RUN:   not %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: SYCL target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check whether an invalid SYCL target is specified:
// RUN:   not %clang -### -fsycl --offload-new-driver -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// RUN:   not %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=x86_64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-REAL-TARGET %s
// CHK-INVALID-REAL-TARGET: error: SYCL target is invalid: 'x86_64'

/// ###########################################################################

/// Check warning for empty -fsycl-targets
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-SYCLTARGETS %s
// CHK-EMPTY-SYCLTARGETS: warning: joined argument expects additional value: '-fsycl-targets='

/// ###########################################################################

/// Check error for no -fsycl --offload-new-driver option
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
// RUN:   not %clang -### -fsycl-device-code-split=bad_value -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-device-code-split %s
// RUN:   not %clang -### -fsycl-link=bad_value -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-BAD-OPT-VALUE -Doption=-fsycl-link %s
// CHK-SYCL-BAD-OPT-VALUE: error: invalid argument 'bad_value' to [[option]]=

/// Check no error for -fsycl-targets with good triple
// RUN:   %clang -### -fsycl-targets=spir-unknown-unknown -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spir64 -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang_cl -### -fsycl-targets=spir-unknown-unknown -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv32-unknown-unknown -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv64-unknown-unknown -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv32 -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// RUN:   %clang -### -fsycl-targets=spirv64 -fsycl --offload-new-driver  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-TARGET %s
// CHK-SYCL-TARGET-NOT: error: SYCL target is invalid

/// -fsycl-device-obj argument check
// RUN: %clang -### -fsycl-device-obj=test -fsycl %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEVICE_OBJ_WARN %s
// RUN: %clang_cl -### -fsycl-device-obj=test -fsycl %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEVICE_OBJ_WARN %s
// DEVICE_OBJ_WARN: warning: ignoring invalid '-fsycl-device-obj' value 'test', using default value 'llvmir' [-Wunused-command-line-argument]

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir64-unknown-unknown  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: SYCL offloading target 'spir64-unknown-unknown' is similar to target 'spir64-unknown-unknown' already specified; will be ignored

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when multiple triples are used.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown,spir-unknown-linux -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR %s
// CHK-FSYCL-COMPILER-AMBIGUOUS-ERROR: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

/// Check -Xsycl-target-frontend triggers error when an option requiring arguments is passed to it.
// RUN:   not %clang -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// RUN:   not %clang_cl -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-frontend -Xsycl-target-frontend -mcpu=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-COMPILER-NESTED-ERROR %s
// CHK-FSYCL-COMPILER-NESTED-ERROR: clang{{.*}} error: invalid -Xsycl-target-frontend argument: '-Xsycl-target-frontend -Xsycl-target-frontend', options requiring arguments are unsupported

/// ###########################################################################

/// Check -Xsycl-target-frontend= accepts triple aliases
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-targets=spir64 -Xsycl-target-frontend=spir64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH1=spir64 -check-prefixes=CHK-UNUSED-ARG-WARNING-1,CHK-TARGET-1 %s
// CHK-UNUSED-ARG-WARNING-1-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET-1: clang{{.*}} "-cc1" "-triple" "[[ARCH1]]-unknown-unknown"{{.*}} "-D" "FOO"
// RUN:   %clang -### -fsycl --offload-new-driver -fsycl-targets=spirv64 -Xsycl-target-frontend=spirv64 -DFOO %s 2>&1 \
// RUN:   | FileCheck -DARCH2=spirv64 -check-prefixes=CHK-UNUSED-ARG-WARNING-2,CHK-TARGET-2 %s
// CHK-UNUSED-ARG-WARNING-2-NOT: clang{{.*}} warning: argument unused during compilation: '-Xsycl-target-frontend={{.*}} -DFOO'
// CHK-TARGET-2: clang{{.*}} "-cc1" "-triple" "[[ARCH2]]-unknown-unknown"{{.*}} "-D" "FOO"

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES %s
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASES: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES: 6: backend, {5}, ir, (device-sycl)
// CHK-PHASES: 7: offload, "device-sycl (spir64-unknown-unknown)" {6}, ir
// CHK-PHASES: 8: clang-offload-packager, {7}, image, (device-sycl)
// CHK-PHASES: 9: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (x86_64{{.*}})" {8}, ir
// CHK-PHASES: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES: 12: clang-linker-wrapper, {11}, image, (host-sycl)

/// ###########################################################################

/// Check the phases graph when using a single target, different from the host.
/// We should have an offload action joining the host compile and device
/// preprocessor and another one joining the device linking outputs to the host
/// action.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spirv64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-2 %s
// RUN:   %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver -fsycl-targets=spirv64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-2 %s
// CHK-PHASES-2: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-2: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-2: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASES-2: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-2: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-2: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-2: 6: backend, {5}, ir, (device-sycl)
// CHK-PHASES-2: 7: offload, "device-sycl (spirv64-unknown-unknown)" {6}, ir
// CHK-PHASES-2: 8: clang-offload-packager, {7}, image, (device-sycl)
// CHK-PHASES-2: 9: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (x86_64{{.*}})" {8}, ir
// CHK-PHASES-2: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES-2: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES-2: 12: clang-linker-wrapper, {11}, image, (host-sycl)

/// ###########################################################################

/// Check the compilation flow to verify that the integrated header is filtered
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -c %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHK-INT-HEADER
// CHK-INT-HEADER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INPUT1:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"
// CHK-INT-HEADER: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include-internal-header" "[[INPUT1]]" "-dependency-filter" "[[INPUT1]]"

/// ###########################################################################

/// Check the phases also add a library to make sure it is treated as input by
/// the device.
// RUN:   %clang -ccc-print-phases -target x86_64-unknown-linux-gnu -lsomelib -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-LIB %s
// CHK-PHASES-LIB: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-LIB: 1: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-LIB: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LIB: 3: compiler, {2}, ir, (host-sycl)
// CHK-PHASES-LIB: 4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-LIB: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LIB: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-LIB: 7: backend, {6}, ir, (device-sycl)
// CHK-PHASES-LIB: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, ir
// CHK-PHASES-LIB: 9: clang-offload-packager, {8}, image, (device-sycl)
// CHK-PHASES-LIB: 10: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {9}, ir
// CHK-PHASES-LIB: 11: backend, {10}, assembler, (host-sycl)
// CHK-PHASES-LIB: 12: assembler, {11}, object, (host-sycl)
// CHK-PHASES-LIB: 13: clang-linker-wrapper, {0, 12}, image, (host-sycl)

/// ###########################################################################

/// Check the phases when using and multiple source files
// RUN:   echo " " > %t.c
// RUN:   %clang -ccc-print-phases -lsomelib -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %s %t.c 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASES-FILES %s
// CHK-PHASES-FILES: 0: input, "somelib", object, (host-sycl)
// CHK-PHASES-FILES: 1: input, "[[INPUT1:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 3: compiler, {2}, ir, (host-sycl)
// CHK-PHASES-FILES: 4: input, "[[INPUT1]]", c++, (device-sycl)
// CHK-PHASES-FILES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-FILES: 7: backend, {6}, ir, (device-sycl)
// CHK-PHASES-FILES: 8: offload, "device-sycl (spir64-unknown-unknown)" {7}, ir
// CHK-PHASES-FILES: 9: clang-offload-packager, {8}, image, (device-sycl)
// CHK-PHASES-FILES: 10: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {9}, ir
// CHK-PHASES-FILES: 11: backend, {10}, assembler, (host-sycl)
// CHK-PHASES-FILES: 12: assembler, {11}, object, (host-sycl)
// CHK-PHASES-FILES: 13: input, "[[INPUT2:.+\.c]]", c++, (host-sycl)
// CHK-PHASES-FILES: 14: preprocessor, {13}, c++-cpp-output, (host-sycl)
// CHK-PHASES-FILES: 15: compiler, {14}, ir, (host-sycl)
// CHK-PHASES-FILES: 16: input, "[[INPUT2]]", c++, (device-sycl)
// CHK-PHASES-FILES: 17: preprocessor, {16}, c++-cpp-output, (device-sycl)
// CHK-PHASES-FILES: 18: compiler, {17}, ir, (device-sycl)
// CHK-PHASES-FILES: 19: backend, {18}, ir, (device-sycl)
// CHK-PHASES-FILES: 20: offload, "device-sycl (spir64-unknown-unknown)" {19}, ir
// CHK-PHASES-FILES: 21: clang-offload-packager, {20}, image, (device-sycl)
// CHK-PHASES-FILES: 22: offload, "host-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (x86_64-unknown-linux-gnu)" {21}, ir
// CHK-PHASES-FILES: 23: backend, {22}, assembler, (host-sycl)
// CHK-PHASES-FILES: 24: assembler, {23}, object, (host-sycl)
// CHK-PHASES-FILES: 25: clang-linker-wrapper, {0, 12, 24}, image, (host-sycl)

/// ###########################################################################

/// Check -fsycl-is-device is passed when compiling for the device.
/// also check for SPIR-V binary creation
// RUN:   %clang -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN:   %clang_cl -### -no-canonical-prefixes -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s

// CHK-FSYCL-IS-DEVICE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check -fsycl-is-device and emitting to .spv when compiling for the device
/// when using -fsycl-device-obj=spirv
// RUN:   %clang -### -fsycl-device-obj=spirv -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s
// RUN:   %clang_cl -### -fsycl-device-obj=spirv -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE-NO-BITCODE %s

// CHK-FSYCL-IS-DEVICE-NO-BITCODE: clang{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}}.c

/// ###########################################################################

/// Check for default linking of -lsycl with -fsycl --offload-new-driver usage
// RUN: %clang -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-SYCL %s
// CHECK-LD-SYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SYCL: "-lsycl"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl --offload-new-driver -nolibsycl -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOLIBSYCL %s
// CHECK-LD-NOLIBSYCL: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOLIBSYCL-NOT: "-lsycl"

/// Check no SYCL runtime is linked with -nostdlib
// RUN: %clang -fsycl --offload-new-driver -nostdlib -target x86_64-unknown-linux-gnu %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LD-NOSTDLIB %s
// CHECK-LD-NOSTDLIB: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-NOSTDLIB-NOT: "-lsycl"

/// Check for default linking of syclN.lib with -fsycl --offload-new-driver usage
// RUN: %clang -fsycl --offload-new-driver -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL %s
// RUN: %clang_cl -fsycl --offload-new-driver %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-CL %s
// CHECK-LINK-SYCL-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-SYCL-CL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"
// CHECK-LINK-SYCL: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check no SYCL runtime is linked with -nolibsycl
// RUN: %clang -fsycl --offload-new-driver -nolibsycl -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL %s
// RUN: %clang_cl -fsycl --offload-new-driver -nolibsycl %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOLIBSYCL-CL %s
// CHECK-LINK-NOLIBSYCL-CL-NOT: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOLIBSYCL: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOLIBSYCL-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check SYCL runtime is linked despite -nostdlib on Windows, this is
/// necessary for the Windows Clang CMake to work
// RUN: %clang -fsycl --offload-new-driver -nostdlib -target x86_64-unknown-windows-msvc %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB %s
// RUN: %clang_cl -fsycl --offload-new-driver -nostdlib %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-NOSTDLIB-CL %s
// CHECK-LINK-NOSTDLIB-CL: "--dependent-lib=sycl{{[0-9]*}}"
// CHECK-LINK-NOSTDLIB: "{{.*}}link{{(.exe)?}}"
// CHECK-LINK-NOSTDLIB: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Check sycld.lib is chosen with /MDd
// RUN:  %clang_cl -fsycl --offload-new-driver /MDd %s -o %t -### 2>&1 | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
/// Check sycld is pulled in when msvcrtd is used
// RUN: %clangxx -fsycl --offload-new-driver -Xclang --dependent-lib=msvcrtd \
// RUN:   -target x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
/// Check sycld.lib is pulled in with -fms-runtime-lib=dll_dbg
// RUN: %clangxx -fsycl --offload-new-driver -fms-runtime-lib=dll_dbg \
// RUN:   -target x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LINK-SYCL-DEBUG %s
// CHECK-LINK-SYCL-DEBUG: "--dependent-lib=sycl{{[0-9]*}}d"
// CHECK-LINK-SYCL-DEBUG-NOT: "-defaultlib:sycl{{[0-9]*}}.lib"

/// Only a single instance of sycld should be pulled in when both the
/// -Xclang --dependent-lib=msvcrtd and -fms-runtime-lib=dll_dbg is used.
// RUN: %clangxx -fsycl --offload-new-driver -fms-runtime-lib=dll_dbg -Xclang \
// RUN:  --dependent-lib=msvcrtd --target=x86_64-unknown-windows-msvc -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-LINK-SYCLD %s
// CHECK-LINK-SYCLD: "--dependent-lib=sycl{{[0-9]*}}d"
// CHECK-LINK-SYCLD-NOT: "--dependent-lib=sycl{{[0-9]*}}d"

/// ###########################################################################

/// Check -Xsycl-target-frontend does not trigger an error when no -fsycl-targets is specified
// RUN:   %clang -### -fsycl --offload-new-driver -Xsycl-target-frontend -DFOO %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-TARGET-ERROR %s
// CHK-NO-FSYCL-TARGET-ERROR-NOT: clang{{.*}} error: cannot deduce implicit triple value for '-Xsycl-target-frontend', specify triple using '-Xsycl-target-frontend=<triple>'

/// ###########################################################################

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS %s
// CHK-TOOLS-OPTS: clang-linker-wrapper{{.*}} "--sycl-backend-compile-options=-DFOO1 -DFOO2"

/// Check for implied options (-g -O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -g -O0 -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Zi -Od -Xsycl-target-backend "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS %s
// CHK-TOOLS-IMPLIED-OPTS: clang-offload-packager{{.*}} {{.*}}compile-opts=-g{{.*}}-DFOO1 -DFOO2"

/// Check for implied options (-O0)
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64 -O0 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-IMPLIED-OPTS-O0 %s
// CHK-TOOLS-IMPLIED-OPTS-O0-NOT: clang-offload-packager{{.*}} {{.*}}compile-opts={{.*}}-cl-opt-disable"

// RUN:   %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -Xsycl-target-linker "-DFOO1 -DFOO2" %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-TOOLS-OPTS2 %s
// CHK-TOOLS-OPTS2: clang-linker-wrapper{{.*}} "--sycl-target-link-options=-DFOO1 -DFOO2"

/// -fsycl-range-rounding settings
///
/// // Check that driver flag is passed to cc1
// RUN: %clang -### -fsycl --offload-new-driver -fsycl-range-rounding=disable %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-DISABLE %s
// RUN: %clang -### -fsycl --offload-new-driver -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-FORCE %s
// RUN: %clang -### -fsycl --offload-new-driver -fsycl-range-rounding=on %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DRIVER-RANGE-ROUNDING-ON %s
// CHK-DRIVER-RANGE-ROUNDING-DISABLE: "-cc1{{.*}}-fsycl-range-rounding=disable"
// CHK-DRIVER-RANGE-ROUNDING-FORCE: "-cc1{{.*}}-fsycl-range-rounding=force"
// CHK-DRIVER-RANGE-ROUNDING-ON: "-cc1{{.*}}-fsycl-range-rounding=on"
///
///
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:        -fsycl-targets=spir64 -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64 -Od %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DISABLE-RANGE-ROUNDING %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:        -O0 -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OVERRIDE-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -Od %s 2>&1 -fsycl-range-rounding=force %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OVERRIDE-RANGE-ROUNDING %s
// CHK-DISABLE-RANGE-ROUNDING: "-fsycl-range-rounding=disable"
// CHK-OVERRIDE-RANGE-ROUNDING: "-fsycl-range-rounding=force"
// CHK-OVERRIDE-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=disable"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:        -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64 -O2 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:        -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// RUN: %clang_cl -### -fsycl --offload-new-driver -fsycl-targets=spir64 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RANGE-ROUNDING %s
// CHK-RANGE-ROUNDING-NOT: "-fsycl-disable-range-rounding"
// CHK-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=disable"
// CHK-RANGE-ROUNDING-NOT: "-fsycl-range-rounding=force"

/// ###########################################################################

/// Verify that triple-boundarch pairs are correct with multi-targetting
// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=nvptx64-nvidia-cuda,spir64 -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 3: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 5: compiler, {4}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 6: backend, {5}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 7: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {6}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 8: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 9: preprocessor, {8}, c++-cpp-output, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 10: compiler, {9}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 11: backend, {10}, ir, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 12: offload, "device-sycl (spir64-unknown-unknown)" {11}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 13: clang-offload-packager, {7, 12}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 14: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (x86_64-unknown-linux-gnu)" {13}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 15: backend, {14}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 16: assembler, {15}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH: 17: clang-linker-wrapper, {16}, image, (host-sycl)

// RUN:  %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:     -fno-sycl-instrument-device-code -fno-sycl-device-lib=all \
// RUN:     -fsycl-targets=nvptx64-nvidia-cuda,spir64_gen \
// RUN:     -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:     -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-PHASE-MULTI-TARG-BOUND-ARCH2 %s
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 3: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 5: compiler, {4}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 6: backend, {5}, ir, (device-sycl, sm_50)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 7: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {6}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 8: input, "[[INPUT]]", c++, (device-sycl, skl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 9: preprocessor, {8}, c++-cpp-output, (device-sycl, skl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 10: compiler, {9}, ir, (device-sycl, skl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 11: backend, {10}, ir, (device-sycl, skl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 12: offload, "device-sycl (spir64_gen-unknown-unknown:skl)" {11}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 13: clang-offload-packager, {7, 12}, image, (device-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 14: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (x86_64-unknown-linux-gnu)" {13}, ir
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 15: backend, {14}, assembler, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 16: assembler, {15}, object, (host-sycl)
// CHK-PHASE-MULTI-TARG-BOUND-ARCH2: 17: clang-linker-wrapper, {16}, image, (host-sycl)

