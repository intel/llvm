///
/// Perform several driver tests for SYCL offloading with -foffload-static-lib
///
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// test behaviors of passing a fat static lib
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t.a %t1_bundle.o
// RUN: llvm-ar cr %t_2.a %t1_bundle.o

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib>
// RUN: touch %t.a
// RUN: touch %t.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-static-lib=%t.a -### %t.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB
// FOFFLOAD_STATIC_LIB: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-outputs=[[OUTLIB:.+\.txt]]"
// FOFFLOAD_STATIC_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB: llvm-link{{.*}} "@[[OUTLIST]]"

/// Use of -foffload-static-lib and -foffload-whole-static-lib are deprecated
// RUN: touch dummy.a
// RUN: %clangxx -fsycl -foffload-static-lib=dummy.a -foffload-whole-static-lib=dummy.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_DEPRECATED
// RUN: %clang_cl -fsycl -foffload-static-lib=dummy.a -foffload-whole-static-lib=dummy.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_DEPRECATED
// FOFFLOAD_STATIC_LIB_DEPRECATED: option '-foffload-whole-static-lib=dummy.a' is deprecated, use 'dummy.a' directly instead

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib> with multiple objects
// RUN: touch %t.a
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: touch %t-3.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-1.o"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-2.o"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-3.o"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-outputs=[[OUTLIB:.+\.txt]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{.*}} "@[[OUTLIST]]"

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib> from source
// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -foffload-static-lib=%t.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC

// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.cpp]]", c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: append-footer, {1}, c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: input, "[[INPUTC]]", c++, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 6: compiler, {5}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {6}, c++-cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 8: compiler, {7}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: backend, {8}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: assembler, {9}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: linker, {0, 10}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: linker, {0, 10}, host_dep_image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: clang-offload-deps, {12}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 14: input, "[[INPUTA]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 15: clang-offload-unbundler, {14}, tempfilelist
// FOFFLOAD_STATIC_LIB_SRC: 16: spirv-to-ir-wrapper, {15}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 17: linker, {6, 13, 16}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 18: sycl-post-link, {17}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 19: file-table-tform, {18}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 20: llvm-spirv, {19}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 21: file-table-tform, {18, 20}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 22: clang-offload-wrapper, {21}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 23: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown)" {22}, image

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// FOFFLOAD_STATIC_LIB_SRC2: clang{{.*}} "-emit-obj" {{.*}} "-o" "[[HOSTOBJ:.+\.o]]"
// FOFFLOAD_STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "-o" "[[HOSTEXE:.+\.out]]"
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-deps{{.*}} "-outputs=[[OUTDEPS:.+\.bc]]" "[[HOSTEXE]]"
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-outputs=[[OUTLIB:.+\.txt]]"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{.*}} "[[OUTDEPS]]" "-o" "[[OUTTEMP:.+\.bc]]"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{.*}} "--only-needed" "[[OUTTEMP]]" "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "[[HOSTOBJ]]"

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC3
// FOFFLOAD_STATIC_LIB_SRC3: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-outputs=[[OUTLIB:.+\.txt]]"
// FOFFLOAD_STATIC_LIB_SRC3: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC3: llvm-link{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC3: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lOpenCL"

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -o output_name -lstdc++ -z relro -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC4
// FOFFLOAD_STATIC_LIB_SRC4: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-outputs=[[OUTLIB:.+\.txt]]"
// FOFFLOAD_STATIC_LIB_SRC4: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIST:.+\.txt]]" "--out-replace=[[OUTLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" "[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC4: llvm-link{{.*}} "@[[OUTLIST]]"
// FOFFLOAD_STATIC_LIB_SRC4: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lstdc++" "-z" "relro"

/// ###########################################################################

/// test behaviors of -foffload-whole-static-lib=<lib>
// RUN: touch %t.a
// RUN: touch %t_2.a
// RUN: touch %t.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-whole-static-lib=%t.a -foffload-whole-static-lib=%t_2.a -### %t.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_WHOLE_STATIC_LIB
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=o" {{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-inputs=[[INPUTA:.+\.a]]" "-outputs=[[OUTLIBA:.+\.txt]]"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIBA]]" "--in-replace=[[OUTLIBA]]" "--out-file-list=[[OUTLISTA:.+\.txt]]" "--out-replace=[[OUTLISTA]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIBA]]" "-o" "[[OUTLISTA]]"
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-inputs=[[INPUTB:.+\.a]]" "-outputs=[[OUTLIBB:.+\.txt]]"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIBB]]" "--in-replace=[[OUTLIBB]]" "--out-file-list=[[OUTLISTB:.+\.txt]]" "--out-replace=[[OUTLISTB]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIBB]]" "-o" "[[OUTLISTB]]"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-link{{.*}} "@[[OUTLISTA]]" "@[[OUTLISTB]]"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-spirv{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-wrapper{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: llc{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: ld{{.*}} "[[INPUTA]]" "[[INPUTB]]"

/// ###########################################################################

/// test behaviors of -foffload-static-lib with no source/object
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -L/dummy/dir -foffload-static-lib=%t.a -### -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_NOSRC_PHASES
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -L/dummy/dir -foffload-whole-static-lib=%t.a -### -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_NOSRC_PHASES
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 1: linker, {0}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 2: linker, {0}, host_dep_image, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 3: clang-offload-deps, {2}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 4: input, "[[INPUTA]]", archive
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 5: clang-offload-unbundler, {4}, tempfilelist
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 6: spirv-to-ir-wrapper, {5}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 7: linker, {3, 6}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 8: sycl-post-link, {7}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 9: file-table-tform, {8}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 10: llvm-spirv, {9}, tempfilelist, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 11: file-table-tform, {8, 10}, tempfiletable, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 12: clang-offload-wrapper, {11}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 13: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {12}, image

/// ###########################################################################

/// test behaviors of -foffload-static-lib with no value
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib= -c %s 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=FOFFLOAD_STATIC_LIB_NOVALUE
// FOFFLOAD_STATIC_LIB_NOVALUE: warning: argument unused during compilation: '-foffload-static-lib='

/// Use of a static archive with various targets should compile and unbundle
// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t.a -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=STATIC_ARCHIVE_UNBUNDLE
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen-unknown-unknown %t.a -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=STATIC_ARCHIVE_UNBUNDLE
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga-unknown-unknown %t.a -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=STATIC_ARCHIVE_UNBUNDLE
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown %t.a -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=STATIC_ARCHIVE_UNBUNDLE
// STATIC_ARCHIVE_UNBUNDLE: clang-offload-bundler{{.*}}
