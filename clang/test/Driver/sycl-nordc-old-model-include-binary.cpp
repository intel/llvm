/// With the old offloading model, the -fno-sycl-rdc device image is already
/// wrapped by clang-offload-wrapper, so it must not also be embedded into the
/// host compilation via -foffload-include-binary.
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   -fsycl-targets=spir64_gen -fno-sycl-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-OLD %s
// CHK-OLD: clang-offload-wrapper
// CHK-OLD-NOT: "-foffload-include-binary"
// CHK-OLD: clang-offload-bundler

/// The new offloading model embeds the finalized device image.
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver -fsycl-targets=spir64_gen -fno-sycl-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NEW %s
// CHK-NEW: "-fsycl-is-host"{{.*}} "-foffload-include-binary"
