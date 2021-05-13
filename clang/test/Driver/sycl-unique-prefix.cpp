/// Test for Unique prefix setting for SYCL compilations
// RUN: touch %t_file1.cpp
// RUN: touch %t_file2.cpp
// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice -c %t_file1.cpp %t_file2.cpp -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PREFIX %s
// CHECK_PREFIX: clang{{.*}} "-triple" "spir64-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-unique-prefix=[[PREFIX1:([A-z0-9]){16}]]"{{.*}} "{{.*}}_file1.cpp"
// CHECK_PREFIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-unique-prefix=[[PREFIX1]]"{{.*}} "{{.*}}_file1.cpp"
// CHECK_PREFIX: clang{{.*}} "-fsycl-unique-prefix=[[PREFIX1]]"{{.*}} "-fsycl-is-host"{{.*}} "{{.*}}_file1.cpp"
// CHECK_PREFIX: clang{{.*}} "-triple" "spir64-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-unique-prefix=[[PREFIX2:([A-z0-9]){16}]]"{{.*}} "{{.*}}_file2.cpp"
// CHECK_PREFIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-unique-prefix=[[PREFIX2]]"{{.*}} "{{.*}}_file2.cpp"
// CHECK_PREFIX: clang{{.*}} "-fsycl-unique-prefix=[[PREFIX2]]"{{.*}} "-fsycl-is-host"{{.*}}  "{{.*}}_file2.cpp"

/// Check for prefix with preprocessed input
// RUN: touch %t.ii
// RUN: %clangxx -fsycl -c %t.ii -### 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_PREFIX_II %s
// CHECK_PREFIX_II: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-unique-prefix=[[PREFIX:([A-z0-9]){16}]]"{{.*}} "{{.*}}.ii"
// CHECK_PREFIX_II: clang{{.*}} "-fsycl-unique-prefix=[[PREFIX]]"{{.*}} "-fsycl-is-host"{{.*}} "{{.*}}.ii"
