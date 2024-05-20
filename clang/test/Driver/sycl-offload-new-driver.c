// REQUIRES: system-linux
/// Verify --offload-new-driver option phases
// RUN:  %clang --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 --offload-new-driver -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFFLOAD-NEW-DRIVER %s
// OFFLOAD-NEW-DRIVER: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// OFFLOAD-NEW-DRIVER: 1: append-footer, {0}, c++, (host-sycl)
// OFFLOAD-NEW-DRIVER: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// OFFLOAD-NEW-DRIVER: 3: compiler, {2}, ir, (host-sycl)
// OFFLOAD-NEW-DRIVER: 4: input, "[[INPUT]]", c++, (device-sycl)
// OFFLOAD-NEW-DRIVER: 5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// OFFLOAD-NEW-DRIVER: 6: compiler, {5}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 7: backend, {6}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 8: offload, "device-sycl (nvptx64-nvidia-cuda)" {7}, ir
// OFFLOAD-NEW-DRIVER: 9: input, "[[INPUT]]", c++, (device-sycl)
// OFFLOAD-NEW-DRIVER: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// OFFLOAD-NEW-DRIVER: 11: compiler, {10}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 12: backend, {11}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 13: offload, "device-sycl (spir64-unknown-unknown)" {12}, ir
// OFFLOAD-NEW-DRIVER: 14: clang-offload-packager, {8, 13}, image, (device-sycl)
// OFFLOAD-NEW-DRIVER: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {14}, ir
// OFFLOAD-NEW-DRIVER: 16: backend, {15}, assembler, (host-sycl)
// OFFLOAD-NEW-DRIVER: 17: assembler, {16}, object, (host-sycl)
// OFFLOAD-NEW-DRIVER: 18: clang-linker-wrapper, {17}, image, (host-sycl)

/// Check the toolflow for SYCL compilation using new offload model
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-FLOW %s
// CHK-FLOW: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" {{.*}} "-fsycl-int-header=[[HEADER:.*]].h" "-fsycl-int-footer=[[FOOTER:.*]].h" {{.*}} "--offload-new-driver" {{.*}} "-o" "[[CC1DEVOUT:.*]]" "-x" "c++" "[[INPUT:.*]]"
// CHK-FLOW-NEXT: clang-offload-packager{{.*}} "-o" "[[PACKOUT:.*]]" "--image=file=[[CC1DEVOUT]],triple=spir64-unknown-unknown,arch=,kind=sycl"
// CHK-FLOW-NEXT: append-file{{.*}} "[[INPUT]]" "--append=[[FOOTER]].h" "--orig-filename=[[INPUT]]" "--output=[[APPENDOUT:.*]]" "--use-include"
// CHK-FLOW-NEXT: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[HEADER]].h" "-dependency-filter" "[[HEADER]].h" {{.*}} "-fsycl-is-host"{{.*}} "-full-main-file-name" "[[INPUT]]" {{.*}} "--offload-new-driver" {{.*}} "-fembed-offload-object=[[PACKOUT]]" {{.*}} "-o" "[[CC1FINALOUT:.*]]" "-x" "c++" "[[APPENDOUT]]"
// CHK-FLOW-NEXT: clang-linker-wrapper{{.*}} "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64"{{.*}} "--linker-path={{.*}}/ld" {{.*}} "[[CC1FINALOUT]]"

/// Verify options passed to clang-linker-wrapper
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          --sysroot=%S/Inputs/SYCL -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS %s
// WRAPPER_OPTIONS: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS-SAME: "-sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o,libsycl-complex-fp64.new.o,libsycl-cmath.new.o,libsycl-cmath-fp64.new.o,libsycl-imf.new.o,libsycl-imf-fp64.new.o,libsycl-imf-bf16.new.o,libsycl-itt-user-wrappers.new.o,libsycl-itt-compiler-wrappers.new.o,libsycl-itt-stubs.new.o"
// WRAPPER_OPTIONS-SAME: "-sycl-device-library-location={{.*}}/lib"

/// Verify phases used to generate SPIR-V instead of LLVM-IR
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-obj=spirv -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix SPIRV_OBJ %s
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-only -fsycl-device-obj=spirv \
// RUN:          -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix SPIRV_OBJ %s
// SPIRV_OBJ: [[#SPVOBJ:]]: input, "{{.*}}", c++, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+1]]: preprocessor, {[[#SPVOBJ]]}, c++-cpp-output, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+2]]: compiler, {[[#SPVOBJ+1]]}, ir, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+3]]: backend, {[[#SPVOBJ+2]]}, ir, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+4]]: llvm-spirv, {[[#SPVOBJ+3]]}, spirv, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+5]]: offload, "device-sycl (spir64-unknown-unknown)" {[[#SPVOBJ+4]]}, {{.*}}

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xspirv-translator -translator-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_TRANSLATOR %s
// WRAPPER_OPTIONS_TRANSLATOR: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS_TRANSLATOR-SAME: "--llvm-spirv-options={{.*}}-translator-opt{{.*}}"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xdevice-post-link -post-link-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_POSTLINK %s
// WRAPPER_OPTIONS_POSTLINK: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS_POSTLINK-SAME: "--sycl-post-link-options=-post-link-opt -O2 -spec-const=native -device-globals -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd"

// -fsycl-device-only behavior
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-only -ccc-print-phases %s 2>&1 \
// RUN   | FileCheck -check-prefix DEVICE_ONLY %s
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          --offload-device-only -ccc-print-phases %s 2>&1 \
// RUN:  | FileCheck -check-prefix DEVICE_ONLY %s
// DEVICE_ONLY: 0: input, "{{.*}}", c++, (device-sycl)
// DEVICE_ONLY: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// DEVICE_ONLY: 2: compiler, {1}, ir, (device-sycl)
// DEVICE_ONLY: 3: backend, {2}, ir, (device-sycl)
// DEVICE_ONLY: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, none
