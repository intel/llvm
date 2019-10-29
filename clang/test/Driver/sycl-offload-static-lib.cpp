///
/// Perform several driver tests for SYCL offloading with -foffload-static-lib
///
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib>
// RUN: touch %t.a
// RUN: touch %t.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-static-lib=%t.a -### %t.o 2>&1 \
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
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_MULTI_O
// FOFFLOAD_STATIC_LIB_MULTI_O: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\-1.o]]" "[[INPUT:.+\-2.o]]" "[[INPUT:.+\-3.o]]" "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_MULTI_O: llvm-link{{.*}} "@{{.*}}"

/// ###########################################################################

/// test behaviors of -foffload-static-lib=<lib> from source
// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC

// FOFFLOAD_STATIC_LIB_SRC: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.cpp]]", c++, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 3: input, "[[INPUTC]]", c++, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 5: compiler, {4}, sycl-header, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-sycldevice)" {5}, c++-cpp-output
// FOFFLOAD_STATIC_LIB_SRC: 7: compiler, {6}, ir, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 8: backend, {7}, assembler, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 9: assembler, {8}, object, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 10: linker, {0, 9}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 11: compiler, {4}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 12: backend, {11}, assembler, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 13: assembler, {12}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 14: input, "[[INPUTA]]", archive
// FOFFLOAD_STATIC_LIB_SRC: 15: clang-offload-unbundler, {14}, object
// FOFFLOAD_STATIC_LIB_SRC: 16: linker, {13, 15}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 17: llvm-spirv, {16}, spirv, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_SRC: 19: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64-unknown-unknown-sycldevice)" {18}, image

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC2
// FOFFLOAD_STATIC_LIB_SRC2: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_SRC2: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_SRC2: llvm-link{{.*}} "@{{.*}}"

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC3
// FOFFLOAD_STATIC_LIB_SRC3: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_SRC3: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_SRC3: llvm-link{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB_SRC3: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lOpenCL"

/// ###########################################################################

// RUN: touch %t.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -foffload-static-lib=%t.a -o output_name -lstdc++ -z relro -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_STATIC_LIB_SRC4
// FOFFLOAD_STATIC_LIB_SRC4: ld{{(.exe)?}}" "-r" "-o" {{.*}} "[[INPUT:.+\.a]]"
// FOFFLOAD_STATIC_LIB_SRC4: clang-offload-bundler{{.*}} "-type=oo"
// FOFFLOAD_STATIC_LIB_SRC4: llvm-link{{.*}} "@{{.*}}"
// FOFFLOAD_STATIC_LIB_SRC4: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lstdc++" "-z" "relro"

/// ###########################################################################

/// test behaviors of -foffload-whole-static-lib=<lib>
// RUN: touch %t.a
// RUN: touch %t.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-whole-static-lib=%t.a -### %t.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FOFFLOAD_WHOLE_STATIC_LIB
// FOFFLOAD_WHOLE_STATIC_LIB: ld{{(.exe)?}}" "-r" "-o" "[[INPUT:.+\.o]]" "{{.*}}crt1.o" "{{.*}}crti.o" "-L/dummy/dir" "[[INPUTO:.+\.o]]" "--whole-archive" "[[INPUTA:.+\.a]]" "--no-whole-archive" "{{.*}}crtn.o"
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=oo" {{.*}} "-inputs=[[INPUT]]"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-link{{.*}} "@{{.*}}"
// FOFFLOAD_WHOLE_STATIC_LIB: llvm-spirv{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: clang-offload-wrapper{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: llc{{.*}}
// FOFFLOAD_WHOLE_STATIC_LIB: ld{{.*}} "[[INPUTA]]" "[[INPUTO]]"

/// ###########################################################################

/// test behaviors of -foffload-static-lib with no source/object
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-static-lib=%t.a -### -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=FOFFLOAD_STATIC_LIB_NOSRC_PHASES,FOFFLOAD_STATIC_LIB_NOSRC_PHASES_1
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir -foffload-whole-static-lib=%t.a -### -ccc-print-phases 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=FOFFLOAD_STATIC_LIB_NOSRC_PHASES,FOFFLOAD_STATIC_LIB_NOSRC_PHASES_2
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 1: linker, {0}, image, (host-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES_1: 2: input, "[[INPUTA]]", archive
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES_2: 2: input, "[[INPUTA]]", wholearchive
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 3: clang-offload-unbundler, {2}, object
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 4: linker, {3}, ir, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 5: llvm-spirv, {4}, spirv, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 6: clang-offload-wrapper, {5}, object, (device-sycl)
// FOFFLOAD_STATIC_LIB_NOSRC_PHASES: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {6}, image
