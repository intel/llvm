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
// OFFLOAD-NEW-DRIVER: 7: backend, {6}, assembler, (device-sycl)
// OFFLOAD-NEW-DRIVER: 8: assembler, {7}, object, (device-sycl)
// OFFLOAD-NEW-DRIVER: 9: offload, "device-sycl (nvptx64-nvidia-cuda)" {8}, object
// OFFLOAD-NEW-DRIVER: 10: input, "[[INPUT]]", c++, (device-sycl)
// OFFLOAD-NEW-DRIVER: 11: preprocessor, {10}, c++-cpp-output, (device-sycl)
// OFFLOAD-NEW-DRIVER: 12: compiler, {11}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 13: backend, {12}, assembler, (device-sycl)
// OFFLOAD-NEW-DRIVER: 14: assembler, {13}, object, (device-sycl)
// OFFLOAD-NEW-DRIVER: 15: offload, "device-sycl (spir64-unknown-unknown)" {14}, object
// OFFLOAD-NEW-DRIVER: 16: clang-offload-packager, {9, 15}, image, (device-sycl)
// OFFLOAD-NEW-DRIVER: 17: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {16}, ir
// OFFLOAD-NEW-DRIVER: 18: backend, {17}, assembler, (host-sycl)
// OFFLOAD-NEW-DRIVER: 19: assembler, {18}, object, (host-sycl)
// OFFLOAD-NEW-DRIVER: 20: clang-linker-wrapper, {19}, image, (host-sycl)

/// Check the toolflow for SYCL compilation using new offload model
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-FLOW %s
// CHK-FLOW: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" {{.*}} "-fsycl-int-header=[[HEADER:.*]].h" "-fsycl-int-footer=[[FOOTER:.*]].h" {{.*}} "--offload-new-driver" {{.*}} "-o" "[[CC1DEVOUT:.*]]" "-x" "c++" "[[INPUT:.*]]"
// CHK-FLOW-NEXT: clang-offload-packager{{.*}} "-o" "[[PACKOUT:.*]]" "--image=file=[[CC1DEVOUT]],triple=spir64-unknown-unknown,arch=,kind=sycl"
// CHK-FLOW-NEXT: append-file{{.*}} "[[INPUT]]" "--append=[[FOOTER]].h" "--orig-filename=[[INPUT]]" "--output=[[APPENDOUT:.*]]" "--use-include"
// CHK-FLOW-NEXT: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[HEADER]].h" "-dependency-filter" "[[HEADER]].h" {{.*}} "-fsycl-is-host"{{.*}} "-full-main-file-name" "[[INPUT]]" {{.*}} "--offload-new-driver" {{.*}} "-fembed-offload-object=[[PACKOUT]]" {{.*}} "-o" "[[CC1FINALOUT:.*]]" "-x" "c++" "[[APPENDOUT]]"
// CHK-FLOW-NEXT: clang-linker-wrapper{{.*}} "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64"{{.*}} "--linker-path={{.*}}/ld" "--" {{.*}} "[[CC1FINALOUT]]"

/// Verify options passed to clang-linker-wrapper
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          --sysroot=%S/Inputs/SYCL -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS %s
// WRAPPER_OPTIONS: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS-SAME: "-sycl-device-libraries=libsycl-crt.o,libsycl-complex.o,libsycl-complex-fp64.o,libsycl-cmath.o,libsycl-cmath-fp64.o,libsycl-imf.o,libsycl-imf-fp64.o,libsycl-imf-bf16.o,libsycl-itt-user-wrappers.o,libsycl-itt-compiler-wrappers.o,libsycl-itt-stubs.o"
// WRAPPER_OPTIONS-SAME: "-sycl-device-library-location={{.*}}/lib"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xspirv-translator -translator-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_TRANSLATOR %s
// WRAPPER_OPTIONS_TRANSLATOR: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS_TRANSLATOR-SAME: "--llvm-spirv-options=-translator-opt"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xdevice-post-link -post-link-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_POSTLINK %s
// WRAPPER_OPTIONS_POSTLINK: clang-linker-wrapper{{.*}} "--triple=spir64"
// WRAPPER_OPTIONS_POSTLINK-SAME: "--sycl-post-link-options=-post-link-opt"
