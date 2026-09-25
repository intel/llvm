// Testing for early device linking.  These tests use -fno-sycl-rdc -c to
// create final device binaries during the compilation step when using -fsycl.
// The finalized device image is merged into the host object with a partial
// link, which makes the object self-contained: it carries the
// __sycl_register_lib constructor and thus can be linked by any host linker.
// Behavior is restricted to the SPIR-V targets.

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
// CREATE_IMAGE: clang{{.*}} "-c" "-o" "[[DEVICE_OBJECT:.+\.o]]" "[[WRAPPER_BC]]"
// CREATE_IMAGE: clang{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OBJECT:.+\.o]]"
// CREATE_IMAGE: ld{{.*}} "-r" "-o" "{{.*}}" "[[DEVICE_OBJECT]]" "[[HOST_OBJECT]]"
 
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu -Xsycl-target-backend \
// RUN:          "-device skl" --sysroot=%S/Inputs/SYCL -ccc-print-phases %s \
// RUN:          -fsycl-instrument-device-code --no-offloadlib 2>&1 \
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
// CREATE_IMAGE_PHASES: 21: partial-linker, {14, 20}, object, (host-sycl)

// Use of -fno-sycl-rdc -c with JIT performs the device link as well, the
// device image is a SPIR-V one to be JIT compiled at run time.
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64 \
// RUN:          --target=x86_64-unknown-linux-gnu -ccc-print-phases %s \
// RUN:          --no-offloadlib 2>&1 \
// RUN:  | FileCheck %s -check-prefix=JIT_ONLY_PHASES
// JIT_ONLY_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// JIT_ONLY_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// JIT_ONLY_PHASES: 2: compiler, {1}, ir, (device-sycl)
// JIT_ONLY_PHASES: 3: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl)
// JIT_ONLY_PHASES: 4: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl)
// JIT_ONLY_PHASES: 5: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl)
// JIT_ONLY_PHASES: 6: linker, {3, 4, 5}, ir, (device-sycl)
// JIT_ONLY_PHASES: 7: linker, {2, 6}, ir, (device-sycl)
// JIT_ONLY_PHASES: 8: sycl-post-link, {7}, tempfiletable, (device-sycl)
// JIT_ONLY_PHASES: 9: file-table-tform, {8}, tempfilelist, (device-sycl)
// JIT_ONLY_PHASES: 10: llvm-spirv, {9}, tempfilelist, (device-sycl)
// JIT_ONLY_PHASES: 11: file-table-tform, {8, 10}, tempfiletable, (device-sycl)
// JIT_ONLY_PHASES: 12: clang-offload-wrapper, {11}, object, (device-sycl)
// JIT_ONLY_PHASES: 13: offload, "device-sycl (spir64-unknown-unknown)" {12}, object
// JIT_ONLY_PHASES: 14: input, "[[INPUT]]", c++, (host-sycl)
// JIT_ONLY_PHASES: 15: preprocessor, {14}, c++-cpp-output, (host-sycl)
// JIT_ONLY_PHASES: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (spir64-unknown-unknown)" {12}, c++-cpp-output
// JIT_ONLY_PHASES: 17: compiler, {16}, ir, (host-sycl)
// JIT_ONLY_PHASES: 18: backend, {17}, assembler, (host-sycl)
// JIT_ONLY_PHASES: 19: assembler, {18}, object, (host-sycl)
// JIT_ONLY_PHASES: 20: partial-linker, {13, 19}, object, (host-sycl)

// Mix and match JIT and AOT phases check.  Expectation is for both to perform
// the early device link, JIT stopping at SPIR-V and AOT going all the way
// down to the target binary.
// RUN: %clangxx -c -fno-sycl-rdc -fsycl --no-offload-new-driver -fsycl-targets=spir64,spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu --sysroot=%S/Inputs/SYCL \
// RUN:          -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:          -fsycl-instrument-device-code -ccc-print-phases %s --no-offloadlib 2>&1 \
// RUN:  | FileCheck %s -check-prefix=JIT_AOT_PHASES
// JIT_AOT_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// JIT_AOT_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// JIT_AOT_PHASES: 2: compiler, {1}, ir, (device-sycl)
// JIT_AOT_PHASES: 3: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 4: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 5: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 6: linker, {3, 4, 5}, ir, (device-sycl)
// JIT_AOT_PHASES: 7: linker, {2, 6}, ir, (device-sycl)
// JIT_AOT_PHASES: 8: sycl-post-link, {7}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 9: file-table-tform, {8}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 10: llvm-spirv, {9}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 11: file-table-tform, {8, 10}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 12: clang-offload-wrapper, {11}, object, (device-sycl)
// JIT_AOT_PHASES: 13: offload, "device-sycl (spir64-unknown-unknown)" {12}, object
// JIT_AOT_PHASES: 14: input, "[[INPUT]]", c++, (device-sycl)
// JIT_AOT_PHASES: 15: preprocessor, {14}, c++-cpp-output, (device-sycl)
// JIT_AOT_PHASES: 16: compiler, {15}, ir, (device-sycl)
// JIT_AOT_PHASES: 17: input, "{{.*}}libsycl-itt-user-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 18: input, "{{.*}}libsycl-itt-compiler-wrappers.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 19: input, "{{.*}}libsycl-itt-stubs.bc", ir, (device-sycl)
// JIT_AOT_PHASES: 20: linker, {17, 18, 19}, ir, (device-sycl)
// JIT_AOT_PHASES: 21: linker, {16, 20}, ir, (device-sycl)
// JIT_AOT_PHASES: 22: sycl-post-link, {21}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 23: file-table-tform, {22}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 24: llvm-spirv, {23}, tempfilelist, (device-sycl)
// JIT_AOT_PHASES: 25: backend-compiler, {24}, image, (device-sycl)
// JIT_AOT_PHASES: 26: file-table-tform, {22, 25}, tempfiletable, (device-sycl)
// JIT_AOT_PHASES: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// JIT_AOT_PHASES: 28: offload, "device-sycl (spir64_gen-unknown-unknown)" {27}, object
// JIT_AOT_PHASES: 29: input, "[[INPUT]]", c++, (host-sycl)
// JIT_AOT_PHASES: 30: preprocessor, {29}, c++-cpp-output, (host-sycl)
// JIT_AOT_PHASES: 31: offload, "host-sycl (x86_64-unknown-linux-gnu)" {30}, "device-sycl (spir64_gen-unknown-unknown)" {27}, c++-cpp-output
// JIT_AOT_PHASES: 32: compiler, {31}, ir, (host-sycl)
// JIT_AOT_PHASES: 33: backend, {32}, assembler, (host-sycl)
// JIT_AOT_PHASES: 34: assembler, {33}, object, (host-sycl)
// JIT_AOT_PHASES: 35: partial-linker, {13, 28, 34}, object, (host-sycl)

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
