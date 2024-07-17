///
/// Perform several driver tests for SYCL offloading on Windows.
///

// REQUIRES: x86-registered-target

/// Test behaviors of using a static library with single object.
// Build the offload library that is used for the tests.
// RUN: echo "void foo() {}" > %t.c
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -c -Fo%t-orig.obj %t.c
// RUN: llvm-ar cr %t-orig.lib %t-orig.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t-orig.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t-orig.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t-orig.obj -### /link %t-orig.lib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o" "-targets=host-x86_64-pc-windows-msvc,sycl-spir64-unknown-unknown" "-input={{.*}}-orig.obj" "-output={{.+}}.{{(o|obj)}}" "-output={{.+}}.{{(o|obj)}}" "-unbundle"
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}-orig.lib" "-output=[[OUTLIB:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB: llvm-link{{(.exe)?}}{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

// RUN: mkdir -p %t_dir
// RUN: llvm-ar cr %t_dir/%basename_t-orig2.lib %t-orig.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t_dir/%basename_t-orig2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB2
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t_dir/%basename_t-orig2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB2
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t-orig.obj -### /link %t_dir/%basename_t-orig2.lib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB2
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t-orig.obj -### /link -libpath:%t_dir %basename_t-orig2.lib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB2
// RUN: env LIB=%t_dir %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-device-lib=all %t-orig.obj -### /link %basename_t-orig2.lib 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB2
// FOFFLOAD_STATIC_LIB2: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o" "-targets=host-x86_64-pc-windows-msvc,sycl-spir64-unknown-unknown" "-input={{.*}}-orig.obj" "-output={{.+}}.{{(o|obj)}}" "-output={{.+}}.{{(o|obj)}}" "-unbundle"
// FOFFLOAD_STATIC_LIB2: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}-orig2.lib" "-output=[[OUTLIB:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_LIB2: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB2: llvm-link{{(.exe)?}}{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB2: link{{(.exe)?}}{{.+}} "{{.*}}-orig2.lib"

/// ###########################################################################

/// Test behaviors of using a static library with multiple objects.
// RUN: touch %t-orig.lib
// RUN: touch %t-1.obj
// RUN: touch %t-2.obj
// RUN: touch %t-3.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t-orig.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t-orig.lib %t-1.obj %t-2.obj %t-3.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-input={{.*}}-1.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-input={{.*}}-2.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-input={{.*}}-3.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}-orig.lib" "-output=[[OUTLIB:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{(.exe)?}}{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

/// ###########################################################################

/// Test behaviors with multiple static libraries.
// RUN: cp %t-orig.lib %t1.lib
// RUN: cp %t-orig.lib %t2.lib
// RUN: touch %t-orig.obj
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t1.lib %t2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_MULTI_LIB
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t1.lib %t2.lib %t-orig.obj -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_MULTI_LIB
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=o"{{.+}} "-input={{.*}}-orig.obj"{{.+}} "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}1.lib" "-output=[[OUTLIB1:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB1]]" "--in-replace=[[OUTLIB1]]" "--out-file-list=[[OUTLIST1:.+\.txt]]" "--out-replace=[[OUTLIST1]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB1]]" "-o" "[[OUTLIST1]]"
// FOFFLOAD_STATIC_MULTI_LIB: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}2.lib" "-output=[[OUTLIB2:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_MULTI_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB2]]" "--in-replace=[[OUTLIB2]]" "--out-file-list=[[OUTLIST2:.+\.txt]]" "--out-replace=[[OUTLIST2]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB2]]" "-o" "[[OUTLIST2:.+\.txt]]"
// FOFFLOAD_STATIC_MULTI_LIB: llvm-link{{(.exe)?}}{{.*}} "@[[OUTLIST1]]" "@[[OUTLIST2]]"
// FOFFLOAD_STATIC_MULTI_LIB: link{{(.exe)?}}{{.+}} "{{.*}}1.lib" "{{.*}}2.lib"

/// ###########################################################################

/// Test behaviors of using static libraries from source.
// RUN: touch %t-orig.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t-orig.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all %t-orig.lib -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC
// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUTLIB:.+\.lib]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.c]]", c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: append-footer, {1}, c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: input, "[[INPUTC]]", c++, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 6: compiler, {5}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 7: offload, "host-sycl (x86_64-pc-windows-msvc)" {3}, "device-sycl (spir64-unknown-unknown)" {6}, c++-cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 8: compiler, {7}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: backend, {8}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: assembler, {9}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: linker, {0, 10}, host_dep_image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: clang-offload-deps, {11}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: input, "[[INPUTLIB]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 14: clang-offload-unbundler, {13}, tempfilelist
// FOFFLOAD_STATIC_LIB_SRC: 15: spirv-to-ir-wrapper, {14}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 16: linker, {6, 12, 15}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 17: sycl-post-link, {16}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 18: file-table-tform, {17}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 19: llvm-spirv, {18}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 20: file-table-tform, {17, 19}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 21: clang-offload-wrapper, {20}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 22: offload, "device-sycl (spir64-unknown-unknown)" {21}, object
// FOFFLOAD_STATIC_LIB_SRC: 23: linker, {0, 10, 22}, image, (host-sycl)

/// ###########################################################################

// RUN: touch %t-orig.lib
// RUN: %clang --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t-orig.lib %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-bundler{{(.exe)?}}{{.+}} "-type=aoo" "-targets=sycl-spir64-{{.+}}" "-input={{.*}}-orig.lib" "-output=[[OUTLIB:.+\.txt]]" "-unbundle"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{(.exe)?}}{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC2: link{{(.exe)?}}{{.+}} "{{.*}}-orig.lib"

// Check for /P behaviors
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver -P %s -### 2>&1 | FileCheck -check-prefix=FSYCL_P %s
// FSYCL_P: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" {{.*}} "-E" {{.*}} "-o" "[[DEVICEPP:.+\.ii]]"
// FSYCL_P: append-file{{.*}} "--output=[[APPEND:.+\.cpp]]"
// FSYCL_P: clang{{.*}} "-cc1" "-triple" "x86_64-pc-windows-msvc{{.*}}" {{.*}} "-E" {{.*}} "-o" "[[HOSTPP:.+\.ii]]"{{.*}} "[[APPEND]]"
// FSYCL_P: clang-offload-bundler{{.*}} "-type=ii" "-targets=sycl-spir64-unknown-unknown,host-x86_64-pc-windows-msvc" {{.*}} "-input=[[DEVICEPP]]" "-input=[[HOSTPP]]"

// RUN: touch %t-orig.lib
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver %t-orig.lib %s -### /link -out:force_out_file 2>&1 \
// RUN:  | FileCheck %s -check-prefix=HOSTDEP_LINK_OVERRIDE
// HOSTDEP_LINK_OVERRIDE: link{{.*}} "-out:[[HOSTDEP_LINK_OUT:.+\.out]]"{{.*}} "-out:force_out_file" "-out:[[HOSTDEP_LINK_OUT]]"
// HOSTDEP_LINK_OVERRIDE: clang-offload-deps{{.*}}
// HOSTDEP_LINK_OVERRIDE: link{{.*}} "-out:[[LINK_OUT:.+\.exe]]"{{.*}} "-out:force_out_file"
// HOSTDEP_LINK_OVERRIDE-NOT: "-out:[[LINK_OUT]]"
