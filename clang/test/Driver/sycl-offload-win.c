///
/// Perform several driver tests for SYCL offloading on Windows.
///

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// Test behaviors of -foffload-static-lib=<lib> with single object.
// Build the offload library that is used for the tests.
// RUN: echo "void foo() {}" > %t.c
// RUN: %clang_cl -fsycl -c -Fo%t-orig.obj %t.c
// RUN: llvm-ar cr %t-orig.lib %t-orig.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all %t-orig.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -fno-sycl-device-lib=all %t-orig.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o" "-targets=host-x86_64-pc-windows-msvc,sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}-orig.obj" "-outputs={{.+}}.{{(o|obj)}},{{.+}}.{{(o|obj)}}" "-unbundle"
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=a" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs={{.*}}-orig.lib" "-outputs=[[OUTLIB:.+\.a]]" "-unbundle"
// FOFFLOAD_STATIC_LIB: llvm-link{{(.exe)?}}{{.*}} "[[OUTLIB]]"
// FOFFLOAD_STATIC_LIB: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

/// ###########################################################################

/// Test behaviors of -foffload-static-lib=<lib> with multiple objects.
// RUN: touch %t-orig.lib
// RUN: touch %t-1.obj
// RUN: touch %t-2.obj
// RUN: touch %t-3.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl %t-orig.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl %t-orig.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs={{.*}}-1.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs={{.*}}-2.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs={{.*}}-3.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=a" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs={{.*}}-orig.lib" "-outputs=[[OUTLIB:.+\.a]]" "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{(.exe)?}}{{.*}} "[[OUTLIB]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

/// ###########################################################################

/// Test behaviors with multiple -foffload-static-lib=<lib> options.
// RUN: cp %t-orig.lib %t1.lib
// RUN: cp %t-orig.lib %t2.lib
// RUN: touch %t-orig.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl %t1.lib %t2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_MULTI_LIB
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl %t1.lib %t2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_MULTI_LIB
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-inputs={{.*}}-orig.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=a" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs={{.*}}1.lib" "-outputs=[[OUTLIB1:.+\.a]]" "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=a" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs={{.*}}2.lib" "-outputs=[[OUTLIB2:.+\.a]]" "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: llvm-link{{(.exe)?}}{{.*}} "[[OUTLIB1]]" "[[OUTLIB2]]"
// FOFFLOAD_STATIC_MULTI_LIB: link{{(.exe)?}}{{.+}} "{{.*}}1.lib" "{{.*}}2.lib"

/// ###########################################################################

/// Test behaviors of -foffload-static-lib=<lib> from source.
// RUN: touch %t-orig.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl-use-footer -fsycl -fno-sycl-device-lib=all %t-orig.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl-use-footer -fsycl -fno-sycl-device-lib=all %t-orig.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUTLIB:.+\.lib]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.c]]", c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: input, "[[INPUTC]]", c++, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: compiler, {4}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 6: offload, "host-sycl (x86_64-pc-windows-msvc)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 7: append-footer, {6}, c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 8: preprocessor, {7}, c++-cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: compiler, {8}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: backend, {9}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: assembler, {10}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: linker, {0, 11}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: linker, {0, 11}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 14: clang-offload-deps, {13}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 15: input, "[[INPUTLIB]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 16: clang-offload-unbundler, {15}, archive
// FOFFLOAD_STATIC_LIB_SRC: 17: linker, {5, 14, 16}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 18: sycl-post-link, {17}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 19: file-table-tform, {18}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 20: llvm-spirv, {19}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 21: file-table-tform, {18, 20}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 22: clang-offload-wrapper, {21}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 23: offload, "host-sycl (x86_64-pc-windows-msvc)" {12}, "device-sycl (spir64-unknown-unknown-sycldevice)" {22}, image

/// ###########################################################################

// RUN: touch %t-orig.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl %t-orig.lib %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -foffload-static-lib=%t-orig.lib %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=a" "-targets=sycl-spir64-{{.+}}-sycldevice" "-inputs={{.*}}-orig.lib" "-outputs=[[OUTLIB:.+\.a]]" "-unbundle"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{(.exe)?}}{{.*}} "[[OUTLIB]]"
// FOFFLOAD_STATIC_LIB_SRC2: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

// Check for /P behaviors
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl -P %s -### 2>&1 | FileCheck -check-prefix=FSYCL_P %s
// FSYCL_P: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown-sycldevice" {{.*}} "-E" {{.*}} "-o" "[[DEVICEPP:.+\.ii]]"
// FSYCL_P: clang{{.*}} "-cc1" "-triple" "x86_64-pc-windows-msvc{{.*}}" {{.*}} "-E" {{.*}} "-o" "[[HOSTPP:.+\.ii]]"
// FSYCL_P: clang-offload-bundler{{.*}} "-type=ii" "-targets=sycl-spir64-unknown-unknown-sycldevice,host-x86_64-pc-windows-msvc" {{.*}} "-inputs=[[DEVICEPP]],[[HOSTPP]]"

