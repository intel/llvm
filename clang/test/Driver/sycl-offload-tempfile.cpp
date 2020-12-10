// UNSUPPORTED: system-windows
// Test temp file cleanup
/// test behaviors of passing a fat static lib

// Build a fat static lib that will be used for all tests
// RUN: echo "void foo(void) {}" > %t1.cpp
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl %t1.cpp -c -o %t1_bundle.o
// RUN: llvm-ar cr %t_lib.a %t1_bundle.o
// RUN: touch %t_obj.o
// RUN: mkdir -p %t_dir
// invoke the compiler overriding output temp location
// RUN: env TMPDIR=%t_dir TEMP=%t_dir TMP=%t_dir                           \
// RUN: %clang -target x86_64-unknown-linux-gnu -### -fsycl %t_obj.o -foffload-static-lib=%t_lib.a 2>&1 | \
// RUN:       FileCheck -DDIRNAME=%t_dir --check-prefix=CHECK-TEMPFILE %s
// RUN: not ls %t_dir/*
// CHECK-TEMPFILE: clang-offload-bundler{{.*}} "-type=aoo" "-targets=sycl-spir64-unknown-unknown-sycldevice" "-inputs={{.*}}" "-outputs=[[DIRNAME]]{{\/|\\}}[[OUTPUT3:.+\.txt]]" "-unbundle"
// CHECK-TEMPFILE: llvm-link{{.*}} "@[[DIRNAME]]{{\/|\\}}[[OUTPUT3]]"

// RUN: mkdir -p %t_dir
// RUN: env TMPDIR=%t_dir TEMP=%t_dir TMP=%t_dir                           \
// RUN: %clang -### -fsycl -fsycl-device-code-split %s 2>&1 | \
// RUN:       FileCheck -DDIRNAME=%t_dir --check-prefix=CHECK-TEMPFILE-SPLIT %s
// RUN: not ls %t_dir/*
// CHECK-TEMPFILE-SPLIT: sycl-post-link{{.*}} "-o" "[[DIRNAME]]{{\/|\\}}[[TABLE:.+\.table]]"
// CHECK-TEMPFILE-SPLIT: file-table-tform{{.*}} "-o" "[[DIRNAME]]{{\/|\\}}{{.+}}.txt"{{.*}} "[[DIRNAME]]{{\/|\\}}[[TABLE]]"
// CHECK-TEMPFILE-SPLIT: llvm-foreach{{.*}} "--out-file-list=[[DIRNAME]]{{\/|\\}}[[OUTPUT:.+\.txt]]" {{.*}}llvm-spirv{{.*}}
