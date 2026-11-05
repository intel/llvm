// Check that with the old offloading model the host objects and archives are
// passed to clang-offload-wrapper so it can define __sycl_registerlib_<id>.

// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver %s 2>&1 | FileCheck %s --check-prefix=SRC
// SRC: clang{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOSTOBJ:[^"]+\.o]]"
// SRC: clang-offload-wrapper{{.*}} "-kind=host" "[[HOSTOBJ]]"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -c %s -o %t.host.o
// RUN: echo 'Content of device file' > %t.dev
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown -input=%t.host.o -input=%t.dev -output=%t.fat.o
// RUN: rm -rf %t.dir && mkdir %t.dir && llvm-ar rc %t.dir/libfat.a %t.fat.o
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver %t.fat.o -L%t.dir -lfat 2>&1 | FileCheck %s --check-prefix=OBJ
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -fsycl-link %t.fat.o -L%t.dir -lfat 2>&1 | FileCheck %s --check-prefix=OBJ
// OBJ: clang-offload-wrapper{{.*}} "-kind=host" "{{[^"]+}}.fat.o" "-kind=host" "{{[^"]+}}libfat.a"

void foo() {}
