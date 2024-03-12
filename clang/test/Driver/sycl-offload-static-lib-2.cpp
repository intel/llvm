///
/// Perform several driver tests for SYCL offloading involving static libs
///
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
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB,STATIC_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_lib.lo -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB,STATIC_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -L/dummy/dir %t_lib.a -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -L/dummy/dir %t_lib.lo -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]" "-input=[[INPUTO:.+\.o]]" "-output=[[HOSTOBJ:.+\.o]]" "-output={{.+\.o}}"
// STATIC_LIB: clang-offload-deps{{.*}} "-targets=[[BUNDLE_TRIPLE]]"
// STATIC_LIB_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}" "-output=[[OUTFILE:.+\.txt]]"
// STATIC_LIB_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}" "-output=[[OUTFILE:.+\.a]]"
// STATIC_LIB_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTFILE]]" "--in-replace=[[OUTFILE]]" "--out-file-list=[[IROUTFILE:.+\.txt]]" "--out-replace=[[IROUTFILE]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTFILE]]" "-o" "[[IROUTFILE]]"
// STATIC_LIB_DEF: llvm-link{{.*}} "@[[IROUTFILE]]"
// STATIC_LIB_NVPTX: llvm-link{{.*}} "[[OUTFILE]]"
// STATIC_LIB: ld{{.*}} "{{.*}}_lib.{{(a|lo)}}" "[[HOSTOBJ]]"

// Test using -l<name> style for passing libraries.
// RUN: mkdir -p %t_dir
// RUN: touch %t_dir/liblin64.so
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L%S/Inputs/SYCL -llin64 -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_L_LIB,STATIC_L_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -static -L%t_dir -L%S/Inputs/SYCL -llin64 -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_L_LIB,STATIC_L_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -Xlinker -Bstatic -L%t_dir -L%S/Inputs/SYCL -llin64 -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_L_LIB,STATIC_L_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -L%S/Inputs/SYCL -llin64 -### %t_obj.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_L_LIB_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_L_LIB: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]" "-input=[[INPUTO:.+\.o]]" "-output=[[HOSTOBJ:.+\.o]]" "-output={{.+\.o}}"
// STATIC_L_LIB: clang-offload-deps{{.*}} "-targets=[[BUNDLE_TRIPLE]]"
// STATIC_L_LIB_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}liblin64.a" "-output=[[OUTFILE:.+\.txt]]"
// STATIC_L_LIB_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}liblin64.a" "-output=[[OUTFILE:.+\.a]]"
// STATIC_L_LIB_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTFILE]]" "--in-replace=[[OUTFILE]]" "--out-file-list=[[IROUTFILE:.+\.txt]]" "--out-replace=[[IROUTFILE]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTFILE]]" "-o" "[[IROUTFILE]]"
// STATIC_L_LIB_DEF: llvm-link{{.*}} "@[[IROUTFILE]]"
// STATIC_L_LIB_NVPTX: llvm-link{{.*}} "[[OUTFILE]]"
// STATIC_L_LIB: ld{{.*}} "-llin64" "[[HOSTOBJ]]"

// non-fat libraries should not trigger the unbundling step.
// presence of shared object should not trigger unbundling step.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L%t_dir -L%S/Inputs/SYCL -llin64 -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=NO_STATIC_UNBUNDLE
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -lc -lm -ldl -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=NO_STATIC_UNBUNDLE
// NO_STATIC_UNBUNDLE-NOT: clang-offload-bundler{{.*}} "-type=aoo" {{.*}} "-input={{.*}}lib{{.*}}.a"

/// ###########################################################################

/// test behaviors of fat static lib with multiple objects
// RUN: touch %t_lib.a
// RUN: touch %t-1.o
// RUN: touch %t-2.o
// RUN: touch %t-3.o
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB_MULTI_O,STATIC_LIB_MULTI_O_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda %t_lib.a -### %t-1.o %t-2.o %t-3.o 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=STATIC_LIB_MULTI_O,STATIC_LIB_MULTI_O_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]" "-input={{.+}}-1.o"
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]" "-input={{.+}}-2.o"
// STATIC_LIB_MULTI_O: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]" "-input={{.+}}-3.o"
// STATIC_LIB_MULTI_O: clang-offload-deps{{.*}} "-targets=[[BUNDLE_TRIPLE]]"
// STATIC_LIB_MULTI_O_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" {{.*}} "-output=[[OUTFILE:.+\.txt]]"
// STATIC_LIB_MULTI_O_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" {{.*}} "-output=[[OUTFILE:.+\.a]]"
// STATIC_LIB_MULTI_O_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTFILE]]" "--in-replace=[[OUTFILE]]" "--out-file-list=[[IROUTFILE:.+\.txt]]" "--out-replace=[[IROUTFILE]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTFILE]]" "-o" "[[IROUTFILE]]"
// STATIC_LIB_MULTI_O_DEF: llvm-link{{.*}} "@[[IROUTFILE]]"
// STATIC_LIB_MULTI_O_NVPTX: llvm-link{{.*}} "[[OUTFILE]]"

/// ###########################################################################

/// test behaviors of fat static lib from source
// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl %t_lib.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=nvptx64-nvidia-cuda -fsycl %t_lib.a -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC-CUDA
// STATIC_LIB_SRC: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// STATIC_LIB_SRC: 1: input, "[[INPUTC:.+\.cpp]]", c++, (host-sycl)
// STATIC_LIB_SRC: 2: append-footer, {1}, c++, (host-sycl)
// STATIC_LIB_SRC: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// STATIC_LIB_SRC: 4: input, "[[INPUTC]]", c++, (device-sycl)
// STATIC_LIB_SRC: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// STATIC_LIB_SRC: 6: compiler, {5}, ir, (device-sycl)
// STATIC_LIB_SRC: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {6}, c++-cpp-output
// STATIC_LIB_SRC: 8: compiler, {7}, ir, (host-sycl)
// STATIC_LIB_SRC: 9: backend, {8}, assembler, (host-sycl)
// STATIC_LIB_SRC: 10: assembler, {9}, object, (host-sycl)
// STATIC_LIB_SRC: 11: linker, {0, 10}, host_dep_image, (host-sycl)
// STATIC_LIB_SRC: 12: clang-offload-deps, {11}, ir, (host-sycl)
// STATIC_LIB_SRC: 13: input, "[[INPUTA]]", archive
// STATIC_LIB_SRC: 14: clang-offload-unbundler, {13}, tempfilelist
// STATIC_LIB_SRC: 15: spirv-to-ir-wrapper, {14}, tempfilelist, (device-sycl)
// STATIC_LIB_SRC: 16: linker, {6, 12, 15}, ir, (device-sycl)
// STATIC_LIB_SRC: 17: sycl-post-link, {16}, tempfiletable, (device-sycl)
// STATIC_LIB_SRC: 18: file-table-tform, {17}, tempfilelist, (device-sycl)
// STATIC_LIB_SRC: 19: llvm-spirv, {18}, tempfilelist, (device-sycl)
// STATIC_LIB_SRC: 20: file-table-tform, {17, 19}, tempfiletable, (device-sycl)
// STATIC_LIB_SRC: 21: clang-offload-wrapper, {20}, object, (device-sycl)
// STATIC_LIB_SRC: 22: offload, "device-sycl (spir64-unknown-unknown)" {21}, object
// STATIC_LIB_SRC: 23: linker, {0, 10, 22}, image, (host-sycl)

// STATIC_LIB_SRC-CUDA: 0: input, "[[INPUTA:.+\.a]]", object, (host-sycl)
// STATIC_LIB_SRC-CUDA: 1: input, "[[INPUTC:.+\.cpp]]", c++, (host-sycl)
// STATIC_LIB_SRC-CUDA: 2: append-footer, {1}, c++, (host-sycl)
// STATIC_LIB_SRC-CUDA: 3: preprocessor, {2}, c++-cpp-output, (host-sycl)
// STATIC_LIB_SRC-CUDA: 4: input, "[[INPUTC]]", c++, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 5: preprocessor, {4}, c++-cpp-output, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 6: compiler, {5}, ir, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {6}, c++-cpp-output
// STATIC_LIB_SRC-CUDA: 8: compiler, {7}, ir, (host-sycl)
// STATIC_LIB_SRC-CUDA: 9: backend, {8}, assembler, (host-sycl)
// STATIC_LIB_SRC-CUDA: 10: assembler, {9}, object, (host-sycl)
// STATIC_LIB_SRC-CUDA: 11: linker, {0, 10}, host_dep_image, (host-sycl)
// STATIC_LIB_SRC-CUDA: 12: clang-offload-deps, {11}, ir, (host-sycl)
// STATIC_LIB_SRC-CUDA: 13: input, "[[INPUTA]]", archive
// STATIC_LIB_SRC-CUDA: 14: clang-offload-unbundler, {13}, archive
// STATIC_LIB_SRC-CUDA: 15: linker, {6, 12, 14}, ir, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 16: sycl-post-link, {15}, ir, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 17: file-table-tform, {16}, ir, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 18: backend, {17}, assembler, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 19: assembler, {18}, object, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 20: linker, {18, 19}, cuda-fatbin, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 21: foreach, {17, 20}, cuda-fatbin, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 22: file-table-tform, {16, 21}, tempfiletable, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 23: clang-offload-wrapper, {22}, object, (device-sycl, sm_50)
// STATIC_LIB_SRC-CUDA: 24: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {23}, object
// STATIC_LIB_SRC-CUDA: 25: linker, {0, 10, 24}, image, (host-sycl)

/// ###########################################################################

// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC2 -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown -DDEPS_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda %t_lib.a -o output_name -lOpenCL -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC2 -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50 -DDEPS_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB_SRC2: clang{{.*}} "-emit-obj" {{.*}} "-o" "[[HOSTOBJ:.+\.o]]"
// STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "-o" "[[HOSTEXE:.+\.out]]" {{.*}}"--unresolved-symbols=ignore-all"
// STATIC_LIB_SRC2: clang-offload-deps{{.*}} "-targets=[[DEPS_TRIPLE]]" "-outputs=[[OUTDEPS:.+\.bc]]" "[[HOSTEXE]]"
// STATIC_LIB_SRC2_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" {{.*}} "-output=[[OUTLIB:.+\.txt]]"
// STATIC_LIB_SRC2_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" {{.*}} "-output=[[OUTLIB:.+\.a]]"
// STATIC_LIB_SRC2_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTLIB]]" "--in-replace=[[OUTLIB]]" "--out-file-list=[[OUTLIBLIST:.+\.txt]]" "--out-replace=[[OUTLIBLIST]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTLIB]]" "-o" [[OUTLIBLIST]]"
// STATIC_LIB_SRC2: llvm-link{{.*}} "[[OUTDEPS]]" "-o" "[[OUTTEMP:.+\.bc]]"
// STATIC_LIB_SRC2_DEF: llvm-link{{.*}} "--only-needed" "[[OUTTEMP]]" "@[[OUTLIBLIST]]"
// STATIC_LIB_SRC2_NVPTX: llvm-link{{.*}} "--only-needed" "[[OUTTEMP]]" "[[OUTLIB]]"
// STATIC_LIB_SRC2: ld{{(.exe)?}}" {{.*}} "[[HOSTOBJ]]"

/// ###########################################################################

// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t_lib.a -o output_name -lstdc++ -z relro -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC3 -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda %t_lib.a -o output_name -lstdc++ -z relro -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_SRC3 -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB_SRC3: clang-offload-bundler{{.*}} "-type=a{{(oo)*}}" "-targets=[[BUNDLE_TRIPLE]]"
// STATIC_LIB_SRC3: llvm-link{{.*}} "{{.*}}"
// STATIC_LIB_SRC3: ld{{(.exe)?}}" {{.*}} "-o" "output_name" {{.*}} "-lstdc++" "-z" "relro"

/// Test device linking behaviors with spir64 and nvptx targets
// RUN: touch %t_lib.a
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda,spir64 %t_lib.a -### %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_MIX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB_MIX: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-nvptx64-nvidia-cuda-sm_50,sycl-spir64-unknown-unknown" {{.*}} "-output=[[NVPTXLIST:.+\.txt]]" "-output=[[SYCLLIST:.+\.txt]]"
// STATIC_LIB_MIX: llvm-link{{.*}} "@[[NVPTXLIST]]"
// STATIC_LIB_MIX: spirv-to-ir-wrapper{{.*}} "[[SYCLLIST]]" "-o" "[[SYCLLINKLIST:.+\.txt]]"
// STATIC_LIB_MIX: llvm-link{{.*}} "@[[SYCLLINKLIST]]"

/// ###########################################################################

/// test behaviors of -Wl,--whole-archive staticlib.a -Wl,--no-whole-archive
/// also test behaviors of -Wl,@arg with the above arguments
// RUN: touch %t_lib.a
// RUN: touch %t_lib_2.a
// RUN: touch %t_obj.o
// RUN: echo "--whole-archive %/t_lib.a %/t_lib_2.a --no-whole-archive" > %t_arg.arg
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_obj.o -Wl,--whole-archive %t_lib.a %t_lib_2.a -Wl,--no-whole-archive -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WHOLE_STATIC_LIB,WHOLE_STATIC_LIB_1,WHOLE_STATIC_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -L/dummy/dir %t_obj.o -Wl,@%/t_arg.arg -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WHOLE_STATIC_LIB,WHOLE_STATIC_LIB_DEF -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -L/dummy/dir %t_obj.o -Wl,--whole-archive %t_lib.a %t_lib_2.a -Wl,--no-whole-archive -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WHOLE_STATIC_LIB,WHOLE_STATIC_LIB_1,WHOLE_STATIC_LIB_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -L/dummy/dir %t_obj.o -Wl,@%/t_arg.arg -### 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=WHOLE_STATIC_LIB,WHOLE_STATIC_LIB_NVPTX -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// WHOLE_STATIC_LIB: clang-offload-bundler{{.*}} "-type=o" "-targets={{.*}},[[BUNDLE_TRIPLE]]"
// WHOLE_STATIC_LIB_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" "-input=[[INPUTA:.+\.a]]" "-output=[[OUTPUTA:.+\.txt]]"
// WHOLE_STATIC_LIB_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTPUTA]]" "--in-replace=[[OUTPUTA]]" "--out-file-list=[[OUTLISTA:.+\.txt]]" "--out-replace=[[OUTLISTA]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTPUTA]]" "-o" "[[OUTLISTA]]"
// WHOLE_STATIC_LIB_DEF: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" "-input=[[INPUTB:.+\.a]]" "-output=[[OUTPUTB:.+\.txt]]"
// WHOLE_STATIC_LIB_DEF: llvm-foreach{{.*}} "--out-ext=txt" "--in-file-list=[[OUTPUTB]]" "--in-replace=[[OUTPUTB]]" "--out-file-list=[[OUTLISTB:.+\.txt]]" "--out-replace=[[OUTLISTB]]" "--" {{.*}}spirv-to-ir-wrapper{{.*}} "[[OUTPUTB]]" "-o" "[[OUTLISTB]]"
// WHOLE_STATIC_LIB_DEF: llvm-link{{.*}} "@[[OUTLISTA]]" "@[[OUTLISTB]]"
// WHOLE_STATIC_LIB_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" "-input=[[INPUTA:.+\.a]]" "-output=[[OUTPUTA:.+\.a]]"
// WHOLE_STATIC_LIB_NVPTX: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" "-input=[[INPUTB:.+\.a]]" "-output=[[OUTPUTB:.+\.a]]"
// WHOLE_STATIC_LIB_NVPTX: llvm-link{{.*}} "[[OUTPUTA]]" "[[OUTPUTB]]"
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
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -L/dummy/dir %t_lib.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC -check-prefix=STATIC_LIB_NOSRC-SPIR -DTARGET=spir64 -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -L/dummy/dir %t_lib.lo -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC -check-prefix=STATIC_LIB_NOSRC-SPIR -DTARGET=spir64 -DBUNDLE_TRIPLE=sycl-spir64-unknown-unknown
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -L/dummy/dir %t_lib.a -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC -check-prefix=STATIC_LIB_NOSRC-CUDA -DTARGET=nvptx64 -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -L/dummy/dir %t_lib.lo -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STATIC_LIB_NOSRC -check-prefix=STATIC_LIB_NOSRC-CUDA -DTARGET=nvptx64 -DBUNDLE_TRIPLE=sycl-nvptx64-nvidia-cuda-sm_50
// STATIC_LIB_NOSRC-SPIR: clang-offload-bundler{{.*}} "-type=aoo" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}_lib.{{(a|lo)}}" "-output=[[DEVICELIB:.+\.txt]]" "-unbundle"
// STATIC_LIB_NOSRC-SPIR: llvm-foreach{{.*}}spirv-to-ir-wrapper{{.*}} "[[DEVICELIB]]" "-o" "[[DEVICELIST:.+\.txt]]"
// STATIC_LIB_NOSRC-SPIR: llvm-link{{.*}} "@[[DEVICELIST]]" "-o" "[[BCFILE:.+\.bc]]"
// STATIC_LIB_NOSRC-CUDA: clang-offload-bundler{{.*}} "-type=a" "-targets=[[BUNDLE_TRIPLE]]" "-input={{.*}}_lib.{{(a|lo)}}" "-output=[[DEVICELIB:.+\.a]]" "-unbundle"
// STATIC_LIB_NOSRC-CUDA: llvm-link{{.*}} "[[DEVICELIB]]" "-o" "[[BCFILE:.+\.bc]]"
// STATIC_LIB_NOSRC: sycl-post-link{{.*}} "-o" "[[TABLE:.+]]" "[[BCFILE]]"
// STATIC_LIB_NOSRC: file-table-tform{{.*}} "-o" "[[LIST:.+]]" "[[TABLE]]"
// STATIC_LIB_NOSRC-SPIR: llvm-foreach{{.*}}llvm-spirv{{.*}} "-o" "[[OBJLIST:.+\.txt]]"{{.*}} "[[LIST]]"
// STATIC_LIB_NOSRC-CUDA: llvm-foreach{{.*}}clang{{.*}} "-o" "[[PTXLIST:.+]]" "-x" "ir" "[[LIST]]"
// STATIC_LIB_NOSRC-CUDA: llvm-foreach{{.*}}ptxas{{.*}} "--output-file" "[[CUBINLIST:.+]]"{{.*}}  "[[PTXLIST]]"
// STATIC_LIB_NOSRC-CUDA: llvm-foreach{{.*}}fatbin{{.*}} "--create" "[[OBJLIST:.+]]"{{.*}} "--image={{.*}}[[PTXLIST]]" "--image={{.*}}[[CUBINLIST]]"
// STATIC_LIB_NOSRC: file-table-tform{{.*}} "-o" "[[TABLE1:.+\.table]]" "[[TABLE]]" "[[OBJLIST]]"
// STATIC_LIB_NOSRC: clang-offload-wrapper{{.*}} "-o=[[BCFILE2:.+\.bc]]" "-host=x86_64-unknown-linux-gnu"{{.*}}"-target=[[TARGET]]" "-kind=sycl" "-batch" "[[TABLE1]]"
// STATIC_LIB_NOSRC: llc{{.*}} "-filetype=obj" "-o" "[[FINALOBJ:.+\.o]]" "[[BCFILE2]]"
// STATIC_LIB_NOSRC: ld{{.*}} "-L/dummy/dir" {{.*}} "{{.*}}_lib.{{(a|lo)}}" "[[FINALOBJ]]"
