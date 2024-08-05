// Testing for early AOT device linking.  These tests use -fno-sycl-rdc
// -c to create final device binaries during the link step when using -fsycl.
// Behavior is restricted to spir64_gen targets for now.

// Create object that contains final device image
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu -Xsycl-target-backend \
// RUN:          "-device skl" --sysroot=%S/Inputs/SYCL -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CREATE_IMAGE
// CREATE_IMAGE: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}} "-o" "[[DEVICE_BC:.+\.bc]]"
// CREATE_IMAGE: llvm-link{{.*}} "-o" "[[LIB_DEVICE_BC:.+\.bc]]"
// CREATE_IMAGE: llvm-link{{.*}} "[[DEVICE_BC]]" "[[LIB_DEVICE_BC]]"{{.*}} "-o" "[[FINAL_DEVICE_BC:.+\.bc]]"
// CREATE_IMAGE: sycl-post-link{{.*}} "-o" "[[POSTLINK_TABLE:.+\.table]]" "[[FINAL_DEVICE_BC]]"
// CREATE_IMAGE: file-table-tform{{.*}} "-o" "[[TFORM_TXT:.+\.txt]]" "[[POSTLINK_TABLE]]"
// CREATE_IMAGE: llvm-spirv{{.*}} "-o" "[[LLVMSPIRV_TXT:.+\.txt]]"{{.*}} "[[TFORM_TXT]]"
// CREATE_IMAGE: ocloc{{.*}} "-output" "[[OCLOC_OUT:.+\.out]]" "-file" "[[LLVMSPIRV_TXT]]"{{.*}} "-device" "skl"
// CREATE_IMAGE: file-table-tform{{.*}} "-o" "[[TFORM_TABLE:.+\.table]]" "[[POSTLINK_TABLE]]" "[[OCLOC_OUT]]"
// CREATE_IMAGE: clang-offload-wrapper{{.*}} "-o=[[WRAPPER_BC:.+\.bc]]"
// CREATE_IMAGE: llc{{.*}} "-o" "[[DEVICE_OBJECT:.+\.o]]" "[[WRAPPER_BC]]"
// CREATE_IMAGE: clang{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OBJECT:.+\.o]]"
// CREATE_IMAGE: clang-offload-bundler{{.*}} "-targets=sycl-spir64_gen_image-unknown-unknown,host-x86_64-unknown-linux-gnu" "-output={{.*}}" "-input=[[DEVICE_OBJECT]]" "-input=[[HOST_OBJECT]]"
 
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu -Xsycl-target-backend \
// RUN:          "-device skl" --sysroot=%S/Inputs/SYCL -ccc-print-phases %s \
// RUN:          -fno-sycl-device-lib=all 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CREATE_IMAGE_PHASES
// CREATE_IMAGE_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// CREATE_IMAGE_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CREATE_IMAGE_PHASES: 2: compiler, {1}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 3: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl)
// CREATE_IMAGE_PHASES: 4: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl)
// CREATE_IMAGE_PHASES: 5: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl)
// CREATE_IMAGE_PHASES: 6: linker, {3, 4, 5}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 7: linker, {2, 6}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 8: sycl-post-link, {7}, tempfiletable, (device-sycl)
// CREATE_IMAGE_PHASES: 9: file-table-tform, {8}, tempfilelist, (device-sycl)
// CREATE_IMAGE_PHASES: 10: llvm-spirv, {9}, tempfilelist, (device-sycl)
// CREATE_IMAGE_PHASES: 11: backend-compiler, {10}, image, (device-sycl)
// CREATE_IMAGE_PHASES: 12: file-table-tform, {8, 11}, tempfiletable, (device-sycl)
// CREATE_IMAGE_PHASES: 13: clang-offload-wrapper, {12}, object, (device-sycl)
// CREATE_IMAGE_PHASES: 14: offload, "device-sycl (spir64_gen-unknown-unknown)" {13}, object
// CREATE_IMAGE_PHASES: 15: input, "[[INPUT]]", c++, (host-sycl)
// CREATE_IMAGE_PHASES: 16: preprocessor, {15}, c++-cpp-output, (host-sycl)
// CREATE_IMAGE_PHASES: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {16}, "device-sycl (spir64_gen-unknown-unknown)" {13}, c++-cpp-output
// CREATE_IMAGE_PHASES: 18: compiler, {17}, ir, (host-sycl)
// CREATE_IMAGE_PHASES: 19: backend, {18}, assembler, (host-sycl)
// CREATE_IMAGE_PHASES: 20: assembler, {19}, object, (host-sycl)
// CREATE_IMAGE_PHASES: 21: clang-offload-bundler, {14, 20}, object, (host-sycl)

// Use of -fno-sycl-rdc -c with non-AOT should not perform the device link.
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64 \
// RUN:          --target=x86_64-unknown-linux-gnu -ccc-print-phases %s \
// RUN:          -fno-sycl-device-lib=all 2>&1 \
// RUN:  | FileCheck %s -check-prefix=JIT_ONLY_PHASES
// JIT_ONLY_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// JIT_ONLY_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// JIT_ONLY_PHASES: 2: compiler, {1}, ir, (device-sycl)
// JIT_ONLY_PHASES: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// JIT_ONLY_PHASES: 4: input, "[[INPUT]]", c++, (host-sycl)
// JIT_ONLY_PHASES: 5: preprocessor, {4}, c++-cpp-output, (host-sycl)
// JIT_ONLY_PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {5}, "device-sycl (spir64-unknown-unknown)" {2}, c++-cpp-output
// JIT_ONLY_PHASES: 7: compiler, {6}, ir, (host-sycl)
// JIT_ONLY_PHASES: 8: backend, {7}, assembler, (host-sycl)
// JIT_ONLY_PHASES: 9: assembler, {8}, object, (host-sycl)
// JIT_ONLY_PHASES: 10: clang-offload-bundler, {3, 9}, object, (host-sycl)

// Mix and match JIT and AOT phases check.  Expectation is for AOT to perform
// early device link, and JIT to just produce the LLVM-IR.
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64,spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:          -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:          -ccc-print-phases %s -fno-sycl-device-lib=all 2>&1 \
// RUN:  | FileCheck %s -check-prefix=JIT_AOT_PHASES
// JIT_AOT_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// JIT_AOT_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// JIT_AOT_PHASES: 2: compiler, {1}, ir, (device-sycl)
// JIT_AOT_PHASES: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// JIT_AOT_PHASES: 4: input, "[[INPUT]]", c++, (device-sycl)
// JIT_AOT_PHASES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// JIT_AOT_PHASES: 6: compiler, {5}, ir, (device-sycl)
// JIT_AOT_PHASES: 7: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 8: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 9: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 10: linker, {7, 8, 9}, ir, (device-sycl)
// JIT_AOT_PHASES: 11: linker, {6, 10}, ir, (device-sycl)
// JIT_AOT_PHASES: 12: sycl-post-link, {11}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 13: file-table-tform, {12}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 14: llvm-spirv, {13}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 15: backend-compiler, {14}, image, (device-sycl)
// JIT_AOT_PHASES: 16: file-table-tform, {12, 15}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// JIT_AOT_PHASES: 18: offload, "device-sycl (spir64_gen-unknown-unknown)" {17}, object
// JIT_AOT_PHASES: 19: input, "[[INPUT]]", c++, (host-sycl)
// JIT_AOT_PHASES: 20: preprocessor, {19}, c++-cpp-output, (host-sycl)
// JIT_AOT_PHASES: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {20}, "device-sycl (spir64_gen-unknown-unknown)" {17}, c++-cpp-output
// JIT_AOT_PHASES: 22: compiler, {21}, ir, (host-sycl)
// JIT_AOT_PHASES: 23: backend, {22}, assembler, (host-sycl)
// JIT_AOT_PHASES: 24: assembler, {23}, object, (host-sycl)
// JIT_AOT_PHASES: 25: clang-offload-bundler, {3, 18, 24}, object, (host-sycl)

// Consume object and library that contain final device images.
// RUN: %clangxx -fsycl --no-offload-new-driver --target=x86_64-unknown-linux-gnu -### \
// RUN:          %S/Inputs/SYCL/objgenimage.o %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CONSUME_OBJ
// CONSUME_OBJ-NOT: linked binaries do not contain expected
// CONSUME_OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen_image-unknown-unknown" "-input={{.*}}objgenimage.o" "-output=[[DEVICE_IMAGE_OBJ:.+\.o]]
// CONSUME_OBJ: ld{{.*}} "[[DEVICE_IMAGE_OBJ]]"

// RUN: %clangxx -fsycl --no-offload-new-driver --target=x86_64-unknown-linux-gnu -### \
// RUN:          %S/Inputs/SYCL/libgenimage.a  %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CONSUME_LIB
// CONSUME_LIB-NOT: linked binaries do not contain expected
// CONSUME_LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64_gen_image-unknown-unknown" "-input={{.*}}libgenimage.a" "-output=[[DEVICE_IMAGE_LIB:.+\.txt]]
// CONSUME_LIB: ld{{.*}} "@[[DEVICE_IMAGE_LIB]]"
