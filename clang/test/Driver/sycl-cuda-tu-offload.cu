// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -target x86_64-unknown-linux-gnu  -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES

// Test the correct placement of the offloading actions for compiling CUDA sources (*.cu) in SYCL.

// DEFAULT-PHASES:                +- 0: input, "{{.*}}", cuda, (device-sycl, sm_80)
// DEFAULT-PHASES:             +- 1: preprocessor, {0}, cuda-cpp-output, (device-sycl, sm_80)
// DEFAULT-PHASES:          +- 2: compiler, {1}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES:       +- 3: offload, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {2}, ir
// DEFAULT-PHASES:       |        +- 4: input, "{{.*}}", cuda, (device-cuda, sm_80)
// DEFAULT-PHASES:       |     +- 5: preprocessor, {4}, cuda-cpp-output, (device-cuda, sm_80)
// DEFAULT-PHASES:       |  +- 6: compiler, {5}, ir, (device-cuda, sm_80)
// DEFAULT-PHASES:       |- 7: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {6}, ir
// DEFAULT-PHASES:    +- 8: linker, {3, 7}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES: +- 9: offload, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {8}, ir
// DEFAULT-PHASES: |                    +- 10: input, "{{.*}}", cuda, (host-cuda-sycl)
// DEFAULT-PHASES: |                 +- 11: append-footer, {10}, cuda, (host-cuda-sycl)
// DEFAULT-PHASES: |              +- 12: preprocessor, {11}, cuda-cpp-output, (host-cuda-sycl)
// DEFAULT-PHASES: |           +- 13: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {12}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {2}, cuda-cpp-output
// DEFAULT-PHASES: |        +- 14: compiler, {13}, ir, (host-cuda-sycl)
// DEFAULT-PHASES: |        |        +- 15: backend, {6}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES: |        |     +- 16: assembler, {15}, object, (device-cuda, sm_80)
// DEFAULT-PHASES: |        |  +- 17: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {16}, object
// DEFAULT-PHASES: |        |  |- 18: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {15}, assembler
// DEFAULT-PHASES: |        |- 19: linker, {17, 18}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES: |     +- 20: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {14}, "device-cuda (nvptx64-nvidia-cuda)" {19}, ir
// DEFAULT-PHASES: |  +- 21: backend, {20}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES: |- 22: assembler, {21}, object, (host-cuda-sycl)
// DEFAULT-PHASES: 23: clang-offload-bundler, {9, 22}, object, (host-cuda-sycl)



// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda -fsycl-libspirv-path=%S/Inputs/SYCL/lib/nvidiacl -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES2

// DEFAULT-PHASES2:                               +- 0: input, "{{.*}}", cuda, (host-cuda-sycl)
// DEFAULT-PHASES2:                            +- 1: append-footer, {0}, cuda, (host-cuda-sycl)
// DEFAULT-PHASES2:                         +- 2: preprocessor, {1}, cuda-cpp-output, (host-cuda-sycl)
// DEFAULT-PHASES2:                         |     +- 3: input, "{{.*}}", cuda, (device-sycl, sm_80)
// DEFAULT-PHASES2:                         |  +- 4: preprocessor, {3}, cuda-cpp-output, (device-sycl, sm_80)
// DEFAULT-PHASES2:                         |- 5: compiler, {4}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:                      +- 6: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {5}, cuda-cpp-output
// DEFAULT-PHASES2:                   +- 7: compiler, {6}, ir, (host-cuda-sycl)
// DEFAULT-PHASES2:                   |                 +- 8: input, "{{.*}}", cuda, (device-cuda, sm_80)
// DEFAULT-PHASES2:                   |              +- 9: preprocessor, {8}, cuda-cpp-output, (device-cuda, sm_80)
// DEFAULT-PHASES2:                   |           +- 10: compiler, {9}, ir, (device-cuda, sm_80)
// DEFAULT-PHASES2:                   |        +- 11: backend, {10}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES2:                   |     +- 12: assembler, {11}, object, (device-cuda, sm_80)
// DEFAULT-PHASES2:                   |  +- 13: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {12}, object
// DEFAULT-PHASES2:                   |  |- 14: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {11}, assembler
// DEFAULT-PHASES2:                   |- 15: linker, {13, 14}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES2:                +- 16: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {7}, "device-cuda (nvptx64-nvidia-cuda)" {15}, ir
// DEFAULT-PHASES2:             +- 17: backend, {16}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES2:          +- 18: assembler, {17}, object, (host-cuda-sycl)
// DEFAULT-PHASES2:       +- 19: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {18}, object
// DEFAULT-PHASES2:    +- 20: linker, {19}, image, (host-cuda-sycl)
// DEFAULT-PHASES2:    |              |- 21: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {10}, ir
// DEFAULT-PHASES2:    |           +- 22: linker, {5, 21}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |           |     +- 23: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 24: clang-offload-unbundler, {23}, object
// DEFAULT-PHASES2:    |           |- 25: offload, " (nvptx64-nvidia-cuda)" {24}, object
// DEFAULT-PHASES2:    |           |     +- 26: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 27: clang-offload-unbundler, {26}, object
// DEFAULT-PHASES2:    |           |- 28: offload, " (nvptx64-nvidia-cuda)" {27}, object
// DEFAULT-PHASES2:    |           |     +- 29: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 30: clang-offload-unbundler, {29}, object
// DEFAULT-PHASES2:    |           |- 31: offload, " (nvptx64-nvidia-cuda)" {30}, object
// DEFAULT-PHASES2:    |           |     +- 32: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 33: clang-offload-unbundler, {32}, object
// DEFAULT-PHASES2:    |           |- 34: offload, " (nvptx64-nvidia-cuda)" {33}, object
// DEFAULT-PHASES2:    |           |     +- 35: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 36: clang-offload-unbundler, {35}, object
// DEFAULT-PHASES2:    |           |- 37: offload, " (nvptx64-nvidia-cuda)" {36}, object
// DEFAULT-PHASES2:    |           |     +- 38: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 39: clang-offload-unbundler, {38}, object
// DEFAULT-PHASES2:    |           |- 40: offload, " (nvptx64-nvidia-cuda)" {39}, object
// DEFAULT-PHASES2:    |           |     +- 41: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 42: clang-offload-unbundler, {41}, object
// DEFAULT-PHASES2:    |           |- 43: offload, " (nvptx64-nvidia-cuda)" {42}, object
// DEFAULT-PHASES2:    |           |     +- 44: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 45: clang-offload-unbundler, {44}, object
// DEFAULT-PHASES2:    |           |- 46: offload, " (nvptx64-nvidia-cuda)" {45}, object
// DEFAULT-PHASES2:    |           |     +- 47: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 48: clang-offload-unbundler, {47}, object
// DEFAULT-PHASES2:    |           |- 49: offload, " (nvptx64-nvidia-cuda)" {48}, object
// DEFAULT-PHASES2:    |           |     +- 50: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 51: clang-offload-unbundler, {50}, object
// DEFAULT-PHASES2:    |           |- 52: offload, " (nvptx64-nvidia-cuda)" {51}, object
// DEFAULT-PHASES2:    |           |     +- 53: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 54: clang-offload-unbundler, {53}, object
// DEFAULT-PHASES2:    |           |- 55: offload, " (nvptx64-nvidia-cuda)" {54}, object
// DEFAULT-PHASES2:    |           |     +- 56: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 57: clang-offload-unbundler, {56}, object
// DEFAULT-PHASES2:    |           |- 58: offload, " (nvptx64-nvidia-cuda)" {57}, object
// DEFAULT-PHASES2:    |           |     +- 59: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 60: clang-offload-unbundler, {59}, object
// DEFAULT-PHASES2:    |           |- 61: offload, " (nvptx64-nvidia-cuda)" {60}, object
// DEFAULT-PHASES2:    |           |     +- 62: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 63: clang-offload-unbundler, {62}, object
// DEFAULT-PHASES2:    |           |- 64: offload, " (nvptx64-nvidia-cuda)" {63}, object
// DEFAULT-PHASES2:    |           |     +- 65: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 66: clang-offload-unbundler, {65}, object
// DEFAULT-PHASES2:    |           |- 67: offload, " (nvptx64-nvidia-cuda)" {66}, object
// DEFAULT-PHASES2:    |           |     +- 68: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 69: clang-offload-unbundler, {68}, object
// DEFAULT-PHASES2:    |           |- 70: offload, " (nvptx64-nvidia-cuda)" {69}, object
// DEFAULT-PHASES2:    |           |     +- 71: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 72: clang-offload-unbundler, {71}, object
// DEFAULT-PHASES2:    |           |- 73: offload, " (nvptx64-nvidia-cuda)" {72}, object
// DEFAULT-PHASES2:    |           |     +- 74: input, "{{.*}}", object
// DEFAULT-PHASES2:    |           |  +- 75: clang-offload-unbundler, {74}, object
// DEFAULT-PHASES2:    |           |- 76: offload, " (nvptx64-nvidia-cuda)" {75}, object
// DEFAULT-PHASES2:    |           |- 77: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |           |- 78: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |        +- 79: linker, {22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 77, 78}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     +- 80: sycl-post-link, {79}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     |  +- 81: file-table-tform, {80}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     |  |  +- 82: backend, {81}, assembler, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     |  |  |- 83: assembler, {82}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     |  |- 84: linker, {82, 83}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |     |- 85: foreach, {81, 84}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |  +- 86: file-table-tform, {80, 85}, tempfiletable, (device-sycl, sm_80)
// DEFAULT-PHASES2:    |- 87: clang-offload-wrapper, {86}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:    88: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {20}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {87}, image
