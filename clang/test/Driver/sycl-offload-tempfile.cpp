// UNSUPPORTED: system-windows
// Test temp file cleanup

// RUN: mkdir -p %t_dir
// RUN: env TMPDIR=%t_dir TEMP=%t_dir TMP=%t_dir                           \
// RUN: %clang -### -fsycl -fsycl-device-code-split %s 2>&1 | \
// RUN:       FileCheck -DDIRNAME=%t_dir --check-prefix=CHECK-TEMPFILE-SPLIT %s
// RUN: not ls %t_dir/*
// CHECK-TEMPFILE-SPLIT: sycl-post-link{{.*}} "-o" "[[DIRNAME]]{{\/|\\}}[[TABLE:.+\.table]]"
// CHECK-TEMPFILE-SPLIT: file-table-tform{{.*}} "-o" "[[DIRNAME]]{{\/|\\}}{{.+}}.txt"{{.*}} "[[DIRNAME]]{{\/|\\}}[[TABLE]]"
// CHECK-TEMPFILE-SPLIT: llvm-foreach{{.*}} "--out-file-list=[[DIRNAME]]{{\/|\\}}[[OUTPUT:.+\.txt]]" {{.*}}llvm-spirv{{.*}}
