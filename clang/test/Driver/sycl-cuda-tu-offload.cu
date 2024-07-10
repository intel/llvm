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
// DEFAULT-PHASES: |              +- 11: preprocessor, {10}, cuda-cpp-output, (host-cuda-sycl)
// DEFAULT-PHASES: |           +- 12: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {11}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {2}, cuda-cpp-output
// DEFAULT-PHASES: |        +- 13: compiler, {12}, ir, (host-cuda-sycl)
// DEFAULT-PHASES: |        |        +- 14: backend, {6}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES: |        |     +- 15: assembler, {14}, object, (device-cuda, sm_80)
// DEFAULT-PHASES: |        |  +- 16: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {15}, object
// DEFAULT-PHASES: |        |  |- 17: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {14}, assembler
// DEFAULT-PHASES: |        |- 18: linker, {16, 17}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES: |     +- 19: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {13}, "device-cuda (nvptx64-nvidia-cuda)" {18}, ir
// DEFAULT-PHASES: |  +- 20: backend, {19}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES: |- 21: assembler, {20}, object, (host-cuda-sycl)
// DEFAULT-PHASES: 22: clang-offload-bundler, {9, 21}, object, (host-cuda-sycl)

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda -fsycl-libspirv-path=%S/Inputs/SYCL/lib/nvidiacl -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES2

// DEFAULT-PHASES2:                               +- 0: input, "{{.*}}", cuda, (host-cuda-sycl)
// DEFAULT-PHASES2:                   +- 1: preprocessor, {0}, cuda-cpp-output, (host-cuda-sycl)
// DEFAULT-PHASES2:                   |     +- 2: input, "{{.*}}", cuda, (device-sycl, sm_80)
// DEFAULT-PHASES2:                   |  +- 3: preprocessor, {2}, cuda-cpp-output, (device-sycl, sm_80)
// DEFAULT-PHASES2:                   |- 4: compiler, {3}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:                +- 5: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {4}, cuda-cpp-output
// DEFAULT-PHASES2:             +- 6: compiler, {5}, ir, (host-cuda-sycl)
// DEFAULT-PHASES2:             |                 +- 7: input, "{{.*}}", cuda, (device-cuda, sm_80)
// DEFAULT-PHASES2:             |              +- 8: preprocessor, {7}, cuda-cpp-output, (device-cuda, sm_80)
// DEFAULT-PHASES2:             |           +- 9: compiler, {8}, ir, (device-cuda, sm_80)
// DEFAULT-PHASES2:             |        +- 10: backend, {9}, assembler, (device-cuda, sm_80)
// DEFAULT-PHASES2:             |     +- 11: assembler, {10}, object, (device-cuda, sm_80)
// DEFAULT-PHASES2:             |  +- 12: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {11}, object
// DEFAULT-PHASES2:             |  |- 13: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {10}, assembler
// DEFAULT-PHASES2:             |- 14: linker, {12, 13}, cuda-fatbin, (device-cuda)
// DEFAULT-PHASES2:          +- 15: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {6}, "device-cuda (nvptx64-nvidia-cuda)" {14}, ir
// DEFAULT-PHASES2:       +- 16: backend, {15}, assembler, (host-cuda-sycl)
// DEFAULT-PHASES2:    +- 17: assembler, {16}, object, (host-cuda-sycl)
// DEFAULT-PHASES2: +- 18: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {17}, object
// DEFAULT-PHASES2: |                 |- 19: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {9}, ir
// DEFAULT-PHASES2: |              +- 20: linker, {4, 19}, ir, (device-sycl, sm_80)
