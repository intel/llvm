// REQUIRES: system-windows
// Test the integration header file name generation.  Name should match the
// actual path name and not the environment variable setting
// RUN: mkdir -p %t_DiRnAmE
// invoke the compiler overriding output temp location
// RUN: env TMP=%t_dirname  \
// RUN: %clang_cl -### -fsycl %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-HEADER %s
// RUN: env TMP=%t_dirname  \
// RUN: %clang -### -fsycl %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-HEADER %s
// CHECK-HEADER: clang{{.*}} "-fsycl-int-header=[[HEADER:.+\.h]]"
// CHECK-HEADER: {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// CHECK-HEADER-NOT: clang{{.*}} "-include" "[[HEADER]]"
// CHECK-HEADER: clang{{.*}} "-include" "{{.*}}_dirname{{.+}}.h"
