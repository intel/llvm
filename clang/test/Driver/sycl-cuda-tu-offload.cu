// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -target x86_64-unknown-linux-gnu  -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES

// Test the correct placement of the offloading actions for compiling CUDA sources (*.cu) in SYCL.

// DEFAULT-PHASES:         +- 0: input, "{{.*}}", cuda, (device-cuda, sm_80)
// DEFAULT-PHASES:      +- 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_80)
// DEFAULT-PHASES:   +- 2: compiler, {1}, ir, (device-cuda, sm_80)
// DEFAULT-PHASES:+- 3: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {2}, ir
// DEFAULT-PHASES:|              +- 4: input, "{{.*}}", cuda, (host-cuda)
// DEFAULT-PHASES:|           +- 5: preprocessor, {4}, cuda-cpp-output, (host-cuda)
// DEFAULT-PHASES:|        +- 6: compiler, {5}, ir, (host-cuda)
// DEFAULT-PHASES:|        |        +- 7: backend, {2}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES:|        |     +- 8: assembler, {7}, object, (device-cuda, sm_80)
// DEFAULT-PHASES:|        |  +- 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {8}, object
// DEFAULT-PHASES:|        |  |- 10: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {7}, assembler
// DEFAULT-PHASES:|        |- 11: linker, {9, 10}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES:|     +- 12: offload, "host-cuda (x86_64-unknown-linux-gnu)" {6}, "device-cuda (nvptx64-nvidia-cuda)" {11}, ir
// DEFAULT-PHASES:|  +- 13: backend, {12}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES:|- 14: assembler, {13}, object, (host-cuda-sycl)
// DEFAULT-PHASES:15: clang-offload-bundler, {3, 14}, object, (host-cuda-sycl)

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda -fsycl-libspirv-path=%S/Inputs/SYCL/lib/nvidiacl -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES2

// DEFAULT-PHASES2:                     +- 0: input, "{{.*}}", cuda, (host-cuda)
// DEFAULT-PHASES2:                  +- 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// DEFAULT-PHASES2:               +- 2: compiler, {1}, ir, (host-cuda)
// DEFAULT-PHASES2:               |                 +- 3: input, "{{.*}}", cuda, (device-cuda, sm_80)
// DEFAULT-PHASES2:               |              +- 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_80)
// DEFAULT-PHASES2:               |           +- 5: compiler, {4}, ir, (device-cuda, sm_80)
// DEFAULT-PHASES2:               |        +- 6: backend, {5}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES2:               |     +- 7: assembler, {6}, object, (device-cuda, sm_80)
// DEFAULT-PHASES2:               |  +- 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {7}, object
// DEFAULT-PHASES2:               |  |- 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {6}, assembler
// DEFAULT-PHASES2:               |- 10: linker, {8, 9}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES2:            +- 11: offload, "host-cuda (x86_64-unknown-linux-gnu)" {2}, "device-cuda (nvptx64-nvidia-cuda)" {10}, ir
// DEFAULT-PHASES2:         +- 12: backend, {11}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES2:      +- 13: assembler, {12}, object, (host-cuda-sycl)
// DEFAULT-PHASES2:   +- 14: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {13}, object
// DEFAULT-PHASES2:+- 15: linker, {14}, image, (host-cuda-sycl)
// DEFAULT-PHASES2:|              +- 16: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {5}, ir
// DEFAULT-PHASES2:|           +- 17: linker, {16}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|           |     +- 18: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 19: clang-offload-unbundler, {18}, object
// DEFAULT-PHASES2:|           |- 20: offload, " (nvptx64-nvidia-cuda)" {19}, object
// DEFAULT-PHASES2:|           |     +- 21: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 22: clang-offload-unbundler, {21}, object
// DEFAULT-PHASES2:|           |- 23: offload, " (nvptx64-nvidia-cuda)" {22}, object
// DEFAULT-PHASES2:|           |     +- 24: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 25: clang-offload-unbundler, {24}, object
// DEFAULT-PHASES2:|           |- 26: offload, " (nvptx64-nvidia-cuda)" {25}, object
// DEFAULT-PHASES2:|           |     +- 27: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 28: clang-offload-unbundler, {27}, object
// DEFAULT-PHASES2:|           |- 29: offload, " (nvptx64-nvidia-cuda)" {28}, object
// DEFAULT-PHASES2:|           |     +- 30: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 31: clang-offload-unbundler, {30}, object
// DEFAULT-PHASES2:|           |- 32: offload, " (nvptx64-nvidia-cuda)" {31}, object
// DEFAULT-PHASES2:|           |     +- 33: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 34: clang-offload-unbundler, {33}, object
// DEFAULT-PHASES2:|           |- 35: offload, " (nvptx64-nvidia-cuda)" {34}, object
// DEFAULT-PHASES2:|           |     +- 36: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 37: clang-offload-unbundler, {36}, object
// DEFAULT-PHASES2:|           |- 38: offload, " (nvptx64-nvidia-cuda)" {37}, object
// DEFAULT-PHASES2:|           |     +- 39: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 40: clang-offload-unbundler, {39}, object
// DEFAULT-PHASES2:|           |- 41: offload, " (nvptx64-nvidia-cuda)" {40}, object
// DEFAULT-PHASES2:|           |     +- 42: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 43: clang-offload-unbundler, {42}, object
// DEFAULT-PHASES2:|           |- 44: offload, " (nvptx64-nvidia-cuda)" {43}, object
// DEFAULT-PHASES2:|           |     +- 45: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 46: clang-offload-unbundler, {45}, object
// DEFAULT-PHASES2:|           |- 47: offload, " (nvptx64-nvidia-cuda)" {46}, object
// DEFAULT-PHASES2:|           |     +- 48: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 49: clang-offload-unbundler, {48}, object
// DEFAULT-PHASES2:|           |- 50: offload, " (nvptx64-nvidia-cuda)" {49}, object
// DEFAULT-PHASES2:|           |     +- 51: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 52: clang-offload-unbundler, {51}, object
// DEFAULT-PHASES2:|           |- 53: offload, " (nvptx64-nvidia-cuda)" {52}, object
// DEFAULT-PHASES2:|           |     +- 54: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 55: clang-offload-unbundler, {54}, object
// DEFAULT-PHASES2:|           |- 56: offload, " (nvptx64-nvidia-cuda)" {55}, object
// DEFAULT-PHASES2:|           |     +- 57: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 58: clang-offload-unbundler, {57}, object
// DEFAULT-PHASES2:|           |- 59: offload, " (nvptx64-nvidia-cuda)" {58}, object
// DEFAULT-PHASES2:|           |     +- 60: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 61: clang-offload-unbundler, {60}, object
// DEFAULT-PHASES2:|           |- 62: offload, " (nvptx64-nvidia-cuda)" {61}, object
// DEFAULT-PHASES2:|           |     +- 63: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 64: clang-offload-unbundler, {63}, object
// DEFAULT-PHASES2:|           |- 65: offload, " (nvptx64-nvidia-cuda)" {64}, object
// DEFAULT-PHASES2:|           |     +- 66: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 67: clang-offload-unbundler, {66}, object
// DEFAULT-PHASES2:|           |- 68: offload, " (nvptx64-nvidia-cuda)" {67}, object
// DEFAULT-PHASES2:|           |     +- 69: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 70: clang-offload-unbundler, {69}, object
// DEFAULT-PHASES2:|           |- 71: offload, " (nvptx64-nvidia-cuda)" {70}, object
// DEFAULT-PHASES2:|           |     +- 72: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 73: clang-offload-unbundler, {72}, object
// DEFAULT-PHASES2:|           |- 74: offload, " (nvptx64-nvidia-cuda)" {73}, object
// DEFAULT-PHASES2:|           |     +- 75: input, "{{.*}}", object
// DEFAULT-PHASES2:|           |  +- 76: clang-offload-unbundler, {75}, object
// DEFAULT-PHASES2:|           |- 77: offload, " (nvptx64-nvidia-cuda)" {76}, object
// DEFAULT-PHASES2:|           |- 78: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|           |- 79: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|        +- 80: linker, {17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 78, 79}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     +- 81: sycl-post-link, {80}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  +- 82: file-table-tform, {81}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |  +- 83: backend, {82}, assembler, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |  |- 84: assembler, {83}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |- 85: linker, {83, 84}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |- 86: foreach, {82, 85}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:|  +- 87: file-table-tform, {81, 86}, tempfiletable, (device-sycl, sm_80)
// DEFAULT-PHASES2:|- 88: clang-offload-wrapper, {87}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:89: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {88}, image
