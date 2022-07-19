// RUN: %clangxx -ccc-print-phases -target x86_64-unknown-linux-gnu  -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES

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

// RUN: %clangxx -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda  -Xsycl-target-backend  --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_80 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES2

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
// DEFAULT-PHASES2:|           +- 16: offload, "device-cuda (nvptx64-nvidia-cuda:sm_80)" {5}, ir
// DEFAULT-PHASES2:|        +- 17: linker, {16}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     +- 18: sycl-post-link, {17}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  +- 19: file-table-tform, {18}, ir, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |  +- 20: backend, {19}, assembler, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |  |- 21: assembler, {20}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |  |- 22: linker, {20, 21}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:|     |- 23: foreach, {19, 22}, cuda-fatbin, (device-sycl, sm_80)
// DEFAULT-PHASES2:|  +- 24: file-table-tform, {18, 23}, tempfiletable, (device-sycl, sm_80)
// DEFAULT-PHASES2:|- 25: clang-offload-wrapper, {24}, object, (device-sycl, sm_80)
// DEFAULT-PHASES2:26: offload, "host-cuda-sycl (x86_64-unknown-linux-gnu)" {15}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {25}, image
