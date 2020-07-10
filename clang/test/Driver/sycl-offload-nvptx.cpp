/// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvcl-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-nvcl-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-Wno-sycl-strict" "-sycl-std=2017" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-feature" "+ptx42"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=nvptx64" "-kind=sycl"{{.*}}
// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-nvcl-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-Wno-sycl-strict" "-sycl-std=2017" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}}  "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-feature" "+ptx42"{{.*}} "-target-sdk-version=[[CUDA_VERSION]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-std=c++11"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 2: input, "{{.*}}", c++, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 4: compiler, {3}, sycl-header, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_50)" {4}, c++-cpp-output
// CHK-PHASES-NO-CC: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-NO-CC: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-NO-CC: 10: compiler, {3}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 11: linker, {10}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 13: backend, {12}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 14: offload, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_50)" {13}, assembler
// CHK-PHASES-NO-CC: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES-NO-CC: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice)" {15}, image

/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "{{.*}}", c++, (device-sycl, sm_35)
// CHK-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES: 4: compiler, {3}, sycl-header, (device-sycl, sm_35)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_35)" {4}, c++-cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: compiler, {3}, ir, (device-sycl, sm_35)
// CHK-PHASES: 11: linker, {10}, ir, (device-sycl, sm_35)
// CHK-PHASES: 12: sycl-post-link, {11}, ir, (device-sycl, sm_35)
// CHK-PHASES: 13: backend, {12}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 14: offload, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_35)" {13}, assembler
// CHK-PHASES: 15: clang-offload-wrapper, {14}, object, (device-sycl)
// CHK-PHASES: 16: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice)" {15}, image

/// Checks when requesting 2 CUDA gpu targets
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" -Xsycl-target-backend "--cuda-gpu-arch=sm_70" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-SM35-70 %s
// CHK-PHASES-SM35-70: 0: input, "[[INPUT:.*]]", c++, (host-sycl)
// CHK-PHASES-SM35-70: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-SM35-70: 2: input, "[[INPUT]]", c++, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 4: compiler, {3}, sycl-header, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_70)" {4}, c++-cpp-output
// CHK-PHASES-SM35-70: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-SM35-70: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-SM35-70: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-SM35-70: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-SM35-70: 10: input, "[[INPUT]]", c++, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 11: preprocessor, {10}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 12: compiler, {11}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 13: linker, {12}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 14: sycl-post-link, {13}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 15: backend, {14}, assembler, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 16: assembler, {15}, object, (device-sycl, sm_35)
// CHK-PHASES-SM35-70: 17: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_35)" {16}, object
// CHK-PHASES-SM35-70: 18: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_35)" {15}, assembler
// CHK-PHASES-SM35-70: 19: compiler, {3}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 20: linker, {19}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 21: sycl-post-link, {20}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 22: backend, {21}, assembler, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 23: assembler, {22}, object, (device-sycl, sm_70)
// CHK-PHASES-SM35-70: 24: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_70)" {23}, object
// CHK-PHASES-SM35-70: 25: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_70)" {22}, assembler
// CHK-PHASES-SM35-70: 26: linker, {17, 18, 24, 25}, cuda-fatbin, (device-sycl)
// CHK-PHASES-SM35-70: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHK-PHASES-SM35-70: 28: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-cuda-sycldevice)" {27}, image

/// Checks when requesting 2 CUDA gpu targets + SPIR
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda-sycldevice,spir64-unknown-unknown-sycldevice \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" -Xsycl-target-backend "--cuda-gpu-arch=sm_70" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-SM35-70-SPIR %s
// CHK-PHASES-SM35-70-SPIR: 0: input, "[[INPUT:.*]]", c++, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 2: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 4: compiler, {3}, sycl-header, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-SM35-70-SPIR: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-SM35-70-SPIR: 10: input, "[[INPUT]]", c++, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 11: preprocessor, {10}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 12: compiler, {11}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 13: linker, {12}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 14: sycl-post-link, {13}, ir, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 15: backend, {14}, assembler, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 16: assembler, {15}, object, (device-sycl, sm_35)
// CHK-PHASES-SM35-70-SPIR: 17: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_35)" {16}, object
// CHK-PHASES-SM35-70-SPIR: 18: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_35)" {15}, assembler
// CHK-PHASES-SM35-70-SPIR: 19: input, "[[INPUT]]", c++, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 20: preprocessor, {19}, c++-cpp-output, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 21: compiler, {20}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 22: linker, {21}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 23: sycl-post-link, {22}, ir, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 24: backend, {23}, assembler, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 25: assembler, {24}, object, (device-sycl, sm_70)
// CHK-PHASES-SM35-70-SPIR: 26: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_70)" {25}, object
// CHK-PHASES-SM35-70-SPIR: 27: offload, "device-sycl (nvptx64-nvidia-cuda-sycldevice:sm_70)" {24}, assembler
// CHK-PHASES-SM35-70-SPIR: 28: linker, {17, 18, 26, 27}, cuda-fatbin, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 29: clang-offload-wrapper, {28}, object, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 30: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 31: linker, {30}, ir, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 32: sycl-post-link, {31}, tempfiletable, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 33: file-table-tform, {32}, tempfilelist, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 34: llvm-spirv, {33}, tempfilelist, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 35: file-table-tform, {32, 34}, tempfiletable, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 36: clang-offload-wrapper, {35}, object, (device-sycl)
// CHK-PHASES-SM35-70-SPIR: 37: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-cuda-sycldevice)" {29}, "device-sycl (spir64-unknown-unknown-sycldevice)" {36}, image

/// Checks invocation when requesting 2 CUDA gpu targets
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" -Xsycl-target-backend "--cuda-gpu-arch=sm_70" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-SM35-70 %s
//      CPP -> BC for sm_35 step
// CHK-INVOKE-SM35-70: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}} "-target-cpu" "sm_35" {{.*}}
//      BC -> PTX for sm_35 step
// CHK-INVOKE-SM35-70: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-S" {{.*}} "-target-cpu" "sm_35" {{.*}} "-o" "[[SM_35_PTX:[^"]*]]"
//      PTX -> CUBIN for sm_35 step
// CHK-INVOKE-SM35-70: "{{.*}}ptxas" {{.*}} "--gpu-name" "sm_35" "--output-file" "[[SM_35_CUBIN:[^"]*]]" "[[SM_35_PTX]]"
//      CPP -> BC for sm_70 step
// CHK-INVOKE-SM35-70: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}} "-target-cpu" "sm_70" {{.*}}
//      BC -> PTX for sm_70 step
// CHK-INVOKE-SM35-70: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-S" {{.*}} "-target-cpu" "sm_70" {{.*}} "-o" "[[SM_70_PTX:[^"]*]]"
//      PTX -> CUBIN for sm_70 step
// CHK-INVOKE-SM35-70: "{{.*}}ptxas" {{.*}} "--gpu-name" "sm_70" "--output-file" "[[SM_70_CUBIN:[^"]*]]" "[[SM_70_PTX]]"
// CHK-INVOKE-SM35-70: "{{.*}}fatbinary" "-64" "--create" "[[FATBIN:[^"]*]]" "--image=profile=sm_35,file=[[SM_35_CUBIN]]" "--image=profile=compute_35,file=[[SM_35_PTX]]" "--image=profile=sm_70,file=[[SM_70_CUBIN]]" "--image=profile=compute_70,file=[[SM_70_PTX]]"
// CHK-INVOKE-SM35-70: "{{.*}}clang-offload-wrapper" {{.*}} "[[FATBIN]]"

/// Checks when requesting 2 CUDA gpu targets + SPIR
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda-sycldevice,spir64-unknown-unknown-sycldevice \
// RUN: -Xsycl-target-backend=nvptx64-nvidia-cuda-sycldevice "--cuda-gpu-arch=sm_35" -Xsycl-target-backend=nvptx64-nvidia-cuda-sycldevice "--cuda-gpu-arch=sm_70" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-SM35-70-SPIR %s
//      CPP -> BC for sm_35 step
// CHK-INVOKE-SM35-70-SPIR: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}} "-target-cpu" "sm_35" {{.*}}
//      BC -> PTX for sm_35 step
// CHK-INVOKE-SM35-70-SPIR: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-S" {{.*}} "-target-cpu" "sm_35" {{.*}} "-o" "[[SM_35_PTX:[^"]*]]"
//      PTX -> CUBIN for sm_35 step
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}ptxas" {{.*}} "--gpu-name" "sm_35" "--output-file" "[[SM_35_CUBIN:[^"]*]]" "[[SM_35_PTX]]"
//      CPP -> BC for sm_70 step
// CHK-INVOKE-SM35-70-SPIR: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-emit-llvm-bc" {{.*}} "-target-cpu" "sm_70" {{.*}}
//      BC -> PTX for sm_70 step
// CHK-INVOKE-SM35-70-SPIR: "-cc1" "-triple" "nvptx64-nvidia-cuda-sycldevice" "-fsycl" "-fsycl-is-device" {{.*}} "-S" {{.*}} "-target-cpu" "sm_70" {{.*}} "-o" "[[SM_70_PTX:[^"]*]]"
//      PTX -> CUBIN for sm_70 step
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}ptxas" {{.*}} "--gpu-name" "sm_70" "--output-file" "[[SM_70_CUBIN:[^"]*]]" "[[SM_70_PTX]]"
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}fatbinary" "-64" "--create" "[[FATBIN:[^"]*]]" "--image=profile=sm_35,file=[[SM_35_CUBIN]]" "--image=profile=compute_35,file=[[SM_35_PTX]]" "--image=profile=sm_70,file=[[SM_70_CUBIN]]" "--image=profile=compute_70,file=[[SM_70_PTX]]"
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}clang-offload-wrapper" "-o=[[CUDA_WRAPPER_BC:[^"]*]]" {{.*}} "[[FATBIN]]"
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}llc" {{.*}} "-o" "[[CUDA_WRAPPER:[^"]*]]" "[[CUDA_WRAPPER_BC]]"
//      SPIR step
// CHK-INVOKE-SM35-70-SPIR: "-cc1" "-triple" "spir64-unknown-unknown-sycldevice" "-fsycl" "-fsycl-is-device"
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}llc" {{.*}} "-o" "[[SPIR_WRAPPER:[^"]*]]"
// CHK-INVOKE-SM35-70-SPIR: "{{.*}}ld" {{.*}} "[[CUDA_WRAPPER]]" "[[SPIR_WRAPPER]]"
