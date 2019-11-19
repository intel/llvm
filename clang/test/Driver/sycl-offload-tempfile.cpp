// UNSUPPORTED: system-windows
// Test temp file cleanup
// RUN: touch %t_obj.o
// RUN: touch %t_lib.a
// RUN: mkdir -p %t_dir
// invoke the compiler overriding output temp location
// RUN: env TMPDIR=%t_dir TEMP=%t_dir TMP=%t_dir                           \
// RUN: %clang -target x86_64-unknown-linux-gnu -### -fsycl %t_obj.o -foffload-static-lib=%t_lib.a 2>&1 | \
// RUN:       FileCheck -DDIRNAME=%t_dir --check-prefix=CHECK-TEMPFILE %s
// RUN: not ls %t_dir/*
// CHECK-TEMPFILE: clang-offload-bundler{{.*}} "-type=oo" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown-sycldevice" "-inputs=[[DIRNAME]]{{.*}}" "-outputs={{.*}},[[DIRNAME]]{{\/|\\}}[[OUTPUT3:.+\.txt]]" "-unbundle"
// CHECK-TEMPFILE: llvm-link{{.*}} "@[[DIRNAME]]{{\/|\\}}[[OUTPUT3]]"
