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
// DEFAULT-PHASES2: +- 19: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {18}, object
// DEFAULT-PHASES2: |                 |- 20: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {10}, ir
// DEFAULT-PHASES2: |              +- 21: linker, {5, 20}, ir, (device-sycl, sm_80)
