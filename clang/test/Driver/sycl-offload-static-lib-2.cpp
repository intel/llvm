///
/// Perform several driver tests for SYCL offloading involving static libs
///
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target

/// ###########################################################################

/// test behaviors of passing a fat static lib
// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: llvm-ar cr %t_lib.lo %t1_bundle.o
// RUN: llvm-ar cr %t_lib_2.a %t1_bundle.o
//
// RUN: touch %t_lib.a
// RUN: touch %t_lib.lo
// RUN: touch %t_obj.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_lib.a -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_lib.lo -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB
// STATIC_LIB: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs=[[INPUTO:.+\.o]]" "-outputs=[[HOSTOBJ:.+\.o]],{{.+\.o}}"
// STATIC_LIB: clang-offload-bundler{{.*}} "-type=a" {{.*}} "-inputs={{.*}}" "-outputs=[[OUTFILE:.+\.a]]"
// STATIC_LIB: llvm-link{{.*}} "[[OUTFILE]]"
// STATIC_LIB: ld{{.*}} "{{.*}}_lib.{{(a|lo)}}" "[[HOSTOBJ]]"

/// ###########################################################################

/// test behaviors of fat static lib with multiple objects
// RUN: touch %t_lib.a
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: touch %t-3.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_MULTI_O
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-1.o"
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-2.o"
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" {{.*}} "-inputs={{.+}}-3.o"
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=a" {{.*}} "-outputs=[[OUTFILE:.+\.a]]"
// STATIC_LIB_MULTI_O: llvm-link{{.*}} "[[OUTFILE]]"

/// ###########################################################################

/// test behaviors of fat static lib from source
// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-device-lib=all -fsycl %t_lib.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC
// STATIC_LIB_SRC: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.cpp]]", c++, (host-sycl)
// STATIC_LIB_SRC: 2: append-footer, {1}, c++, (host-sycl)
// STATIC_LIB_SRC: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// STATIC_LIB_SRC: 4: input, "[[INPUTC]]", c++, (device-sycl)
// STATIC_LIB_SRC: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// STATIC_LIB_SRC: 6: compiler, {5}, ir, (device-sycl)
// STATIC_LIB_SRC: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {6}, c++-cpp-output
// STATIC_LIB_SRC: 8: compiler, {7}, ir, (host-sycl)
// STATIC_LIB_SRC: 9: backend, {8}, assembler, (host-sycl)
// STATIC_LIB_SRC: 10: assembler, {9}, object, (host-sycl)
// STATIC_LIB_SRC: 11: linker, {0, 10}, image, (host-sycl)
// STATIC_LIB_SRC: 12: linker, {0, 10}, host_dep_image, (host-sycl)
// STATIC_LIB_SRC: 13: clang-offload-deps, {12}, ir, (host-sycl)
// STATIC_LIB_SRC: 14: input, "[[INPUTA]]", archive
// STATIC_LIB_SRC: 15: clang-offload-unbundler, {14}, archive
// STATIC_LIB_SRC: 16: linker, {6, 13, 15}, ir, (device-sycl)
// STATIC_LIB_SRC: 17: sycl-post-link, {16}, tempfiletable, (device-sycl)
// STATIC_LIB_SRC: 18: file-table-tform, {17}, tempfilelist, (device-sycl)
// STATIC_LIB_SRC: 19: llvm-spirv, {18}, tempfilelist, (device-sycl)
// STATIC_LIB_SRC: 20: file-table-tform, {17, 19}, tempfiletable, (device-sycl)
// STATIC_LIB_SRC: 21: clang-offload-wrapper, {20}, object, (device-sycl)
// STATIC_LIB_SRC: 22: offload, "host-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (spir64-unknown-unknown-sycldevice)" {21}, image

/// ###########################################################################

// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC2
// STATIC_LIB_SRC2: clang{{.*}} "-emit-obj" {{.*}} "-o" "[[HOSTOBJ:.+\.o]]"
// STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "-o" "[[HOSTEXE:.+\.out]]"
// STATIC_LIB_SRC2: clang-offload-deps{{.*}} "-outputs=[[OUTDEPS:.+\.bc]]" "[[HOSTEXE]]"
// STATIC_LIB_SRC2: clang-offload-bundler{{.*}} "-type=a" {{.*}} "-outputs=[[OUTLIB:.+\.a]]"
// STATIC_LIB_SRC2: llvm-link{{.*}} "[[OUTDEPS]]" "-o" "[[OUTTEMP:.+\.bc]]"
// STATIC_LIB_SRC2: llvm-link{{.*}} "--only-needed" "[[OUTTEMP]]" "[[OUTLIB]]"
// STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "[[HOSTOBJ]]"

/// ###########################################################################

// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -o output_name -lstdc++ -z relro -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC3
// STATIC_LIB_SRC3: clang-offload-bundler{{.*}} "-type=a"
// STATIC_LIB_SRC3: llvm-link{{.*}} "{{.*}}"
// STATIC_LIB_SRC3: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lstdc++" "-z" "relro"

/// ###########################################################################

/// test behaviors of -Wl,--whole-archive staticlib.a -Wl,--no-whole-archive
/// also test behaviors of -Wl,@arg with the above arguments
// RUN: touch %t_lib.a
// RUN: touch %t_lib_2.a
// RUN: touch %t_obj.o
// RUN: echo "--whole-archive %/t_lib.a %/t_lib_2.a --no-whole-archive" > %t_arg.arg
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_obj.o -Wl,--whole-archive %t_lib.a %t_lib_2.a -Wl,--no-whole-archive -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WHOLE_STATIC_LIB,WHOLE_STATIC_LIB_1
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_obj.o -Wl,@%/t_arg.arg -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=WHOLE_STATIC_LIB
// WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=o" {{.*}}
// WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=a" {{.*}} "-inputs=[[INPUTA:.+\.a]]" "-outputs=[[OUTPUTA:.+\.a]]"
// WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=a" {{.*}} "-inputs=[[INPUTB:.+\.a]]" "-outputs=[[OUTPUTB:.+\.a]]"
// WHOLE_STATIC_LIB: llvm-link{{.*}} "[[OUTPUTA]]" "[[OUTPUTB]]"
// WHOLE_STATIC_LIB: llvm-spirv{{.*}}
// WHOLE_STATIC_LIB: clang-offload-wrapper{{.*}}
// WHOLE_STATIC_LIB: llc{{.*}}
// WHOLE_STATIC_LIB_1: ld{{.*}} "--whole-archive" "[[INPUTA]]" "[[INPUTB]]" "--no-whole-archive"

/// test behaviors for special case handling of -z and -rpath
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -z anystring -L/dummy/dir %t_obj.o -Wl,-rpath,nopass -Wl,-z,nopass %t_lib.a %t_lib_2.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WL_CHECK
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -z anystring -L/dummy/dir %t_obj.o -Wl,-rpath -Wl,nopass -Xlinker -z -Xlinker nopass %t_lib.a %t_lib_2.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WL_CHECK
// WL_CHECK-NOT: ld{{.*}} "-r" {{.*}} "anystring" {{.*}} "nopass"
// WL_CHECK: ld{{.*}} "-z" "anystring" {{.*}} "-rpath" "nopass" "-z" "nopass"

/// ###########################################################################

/// test behaviors of static lib with no source/object
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -L/dummy/dir %t_lib.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all -L/dummy/dir %t_lib.lo -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC
// STATIC_LIB_NOSRC: clang-offload-bundler{{.*}} "-type=a" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}_lib.{{(a|lo)}}" "-outputs=[[DEVICELIB:.+\.a]]" "-unbundle"
// STATIC_LIB_NOSRC: llvm-link{{.*}} "[[DEVICELIB]]" "-o" "[[BCFILE:.+\.bc]]"
// STATIC_LIB_NOSRC: sycl-post-link{{.*}} "-o" "[[TABLE:.+\.table]]" "[[BCFILE]]"
// STATIC_LIB_NOSRC: file-table-tform{{.*}} "-o" "[[LIST:.+\.txt]]" "[[TABLE]]"
// STATIC_LIB_NOSRC: llvm-foreach{{.*}}llvm-spirv{{.*}} "-o" "[[SPVLIST:.+\.txt]]"{{.*}}  "[[LIST]]"
// STATIC_LIB_NOSRC: file-table-tform{{.*}} "-o" "[[TABLE1:.+\.table]]" "[[TABLE]]" "[[SPVLIST]]"
// STATIC_LIB_NOSRC: clang-offload-wrapper{{.*}} "-o=[[BCFILE2:.+\.bc]]" "-host=x86_64-unknown-linux-gnu" "-target=spir64" "-kind=sycl" "-batch" "[[TABLE1]]"
// STATIC_LIB_NOSRC: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJ:.+\.o]]" "[[BCFILE2]]"
// STATIC_LIB_NOSRC: ld{{.*}} "-L/dummy/dir" {{.*}} "{{.*}}_lib.{{(a|lo)}}" "[[FINALOBJ]]"
