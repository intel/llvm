// RUN: %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   -fno-bundle-offload-arch -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BUNDLE-TAG-OBJ %s
//
// CHK-BUNDLE-TAG-OBJ-NOT: clang-offload-bundler{{.*}}-targets=sycl-nvptx64-nvidia-cuda-sm_"

// RUN: %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   -fno-bundle-offload-arch %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BUNDLE-TAG %s
//
// CHK-BUNDLE-TAG-NOT: clang-offload-bundler{{.*}}-targets=sycl-nvptx64-nvidia-cuda-sm_"

void func(){};
