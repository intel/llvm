// Testing for early AOT device linking.  These tests use -ftarget-device-link
// -c to create final device binaries during the link step when using -fsycl.
// Behavior is restricted to spir64_gen targets for now.

// Create object that contains final device image
// RUN: %clangxx -c -ftarget-device-link -fsycl -fsycl-targets=spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu -Xsycl-target-backend \
// RUN:          "-device skl" --sysroot=%S/Inputs/SYCL -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CREATE_IMAGE
// CREATE_IMAGE: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}} "-o" "[[DEVICE_BC:.+\.bc]]"
// CREATE_IMAGE: llvm-link{{.*}} "[[DEVICE_BC]]"{{.*}} "-o" "[[DEVICE_BC2:.+\.bc]]"
// CREATE_IMAGE: llvm-link{{.*}} "-only-needed"{{.*}} "[[DEVICE_BC2]]"{{.*}} "-o" "[[DEVICE_BC3:.+\.bc]]"
// CREATE_IMAGE: sycl-post-link{{.*}} "-o" "[[POSTLINK_TABLE:.+\.table]]" "[[DEVICE_BC3]]"
// CREATE_IMAGE: file-table-tform{{.*}} "-o" "[[TFORM_TXT:.+\.txt]]" "[[POSTLINK_TABLE]]"
// CREATE_IMAGE: llvm-spirv{{.*}} "-o" "[[LLVMSPIRV_TXT:.+\.txt]]"{{.*}} "[[TFORM_TXT]]"
// CREATE_IMAGE: ocloc{{.*}} "-output" "[[OCLOC_OUT:.+\.out]]" "-file" "[[LLVMSPIRV_TXT]]"{{.*}} "-device" "skl"
// CREATE_IMAGE: file-table-tform{{.*}} "-o" "[[TFORM_TABLE:.+\.table]]" "[[POSTLINK_TABLE]]" "[[OCLOC_OUT]]"
// CREATE_IMAGE: clang-offload-wrapper{{.*}} "-o=[[WRAPPER_BC:.+\.bc]]"
// CREATE_IMAGE: llc{{.*}} "-o" "[[DEVICE_OBJECT:.+\.o]]" "[[WRAPPER_BC]]"
// CREATE_IMAGE: append-file{{.*}} "--output=[[APPEND_SOURCE:.+\.cpp]]
// CREATE_IMAGE: clang{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OBJECT:.+\.o]]"{{.*}} "[[APPEND_SOURCE]]"
// CREATE_IMAGE: clang-offload-bundler{{.*}} "-targets=sycl-spir64_gen_image-unknown-unknown,host-x86_64-unknown-linux-gnu" "-output={{.*}}" "-input=[[DEVICE_OBJECT]]" "-input=[[HOST_OBJECT]]"
 
// RUN: %clangxx -c -ftarget-device-link -fsycl -fsycl-targets=spir64_gen \
// RUN:          --target=x86_64-unknown-linux-gnu -Xsycl-target-backend \
// RUN:          "-device skl" --sysroot=%S/Inputs/SYCL -ccc-print-phases %s \
// RUN:          -fno-sycl-device-lib=all 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CREATE_IMAGE_PHASES
// CREATE_IMAGE_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// CREATE_IMAGE_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CREATE_IMAGE_PHASES: 2: compiler, {1}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 3: linker, {2}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 4: input, "{{.*libsycl-itt-user-wrappers.o.*}}", object
// CREATE_IMAGE_PHASES: 5: clang-offload-unbundler, {4}, object
// CREATE_IMAGE_PHASES: 6: offload, " (spir64_gen-unknown-unknown)" {5}, object
// CREATE_IMAGE_PHASES: 7: input, "{{.*libsycl-itt-compiler-wrappers.o.*}}", object
// CREATE_IMAGE_PHASES: 8: clang-offload-unbundler, {7}, object
// CREATE_IMAGE_PHASES: 9: offload, " (spir64_gen-unknown-unknown)" {8}, object
// CREATE_IMAGE_PHASES: 10: input, "{{.*libsycl-itt-stubs.o.*}}", object
// CREATE_IMAGE_PHASES: 11: clang-offload-unbundler, {10}, object
// CREATE_IMAGE_PHASES: 12: offload, " (spir64_gen-unknown-unknown)" {11}, object
// CREATE_IMAGE_PHASES: 13: linker, {3, 6, 9, 12}, ir, (device-sycl)
// CREATE_IMAGE_PHASES: 14: sycl-post-link, {13}, tempfiletable, (device-sycl)
// CREATE_IMAGE_PHASES: 15: file-table-tform, {14}, tempfilelist, (device-sycl)
// CREATE_IMAGE_PHASES: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CREATE_IMAGE_PHASES: 17: backend-compiler, {16}, image, (device-sycl)
// CREATE_IMAGE_PHASES: 18: file-table-tform, {14, 17}, tempfiletable, (device-sycl)
// CREATE_IMAGE_PHASES: 19: clang-offload-wrapper, {18}, object, (device-sycl)
// CREATE_IMAGE_PHASES: 20: offload, "device-sycl (spir64_gen-unknown-unknown)" {19}, object
// CREATE_IMAGE_PHASES: 21: offload, "device-sycl (spir64_gen-unknown-unknown)" {20}, object
// CREATE_IMAGE_PHASES: 22: input, "[[INPUT]]", c++, (host-sycl)
// CREATE_IMAGE_PHASES: 23: append-footer, {22}, c++, (host-sycl)
// CREATE_IMAGE_PHASES: 24: preprocessor, {23}, c++-cpp-output, (host-sycl)
// CREATE_IMAGE_PHASES: 25: offload, "host-sycl (x86_64-unknown-linux-gnu)" {24}, "device-sycl (spir64_gen-unknown-unknown)" {20}, c++-cpp-output
// CREATE_IMAGE_PHASES: 26: compiler, {25}, ir, (host-sycl)
// CREATE_IMAGE_PHASES: 27: backend, {26}, assembler, (host-sycl)
// CREATE_IMAGE_PHASES: 28: assembler, {27}, object, (host-sycl)
// CREATE_IMAGE_PHASES: 29: clang-offload-bundler, {21, 28}, object, (host-sycl)

// Consume object and library that contain final device images.
// RUN: %clangxx -fsycl --target=x86_64-unknown-linux-gnu -### \
// RUN:          %S/Inputs/SYCL/objgenimage.o %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CONSUME_OBJ
// CONSUME_OBJ: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64_gen_image-unknown-unknown" "-input={{.*}}objgenimage.o" "-output=[[DEVICE_IMAGE_OBJ:.+\.o]]
// CONSUME_OBJ: ld{{.*}} "[[DEVICE_IMAGE_OBJ]]"

// RUN: %clangxx -fsycl --target=x86_64-unknown-linux-gnu -### \
// RUN:          %S/Inputs/SYCL/libgenimage.a  %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CONSUME_LIB
// CONSUME_LIB: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64_gen_image-unknown-unknown" "-input={{.*}}libgenimage.a" "-output=[[DEVICE_IMAGE_LIB:.+\.txt]]
// CONSUME_LIB: ld{{.*}} "@[[DEVICE_IMAGE_LIB]]"
