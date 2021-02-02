/// Check "-aux-target-cpu" and "target-cpu" are passed when compiling for SYCL offload device and host codes:
//  RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OFFLOAD %s
//  CHECK-OFFLOAD: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  CHECK-OFFLOAD-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]"
//  CHECK-OFFLOAD-NEXT: clang{{.*}} "-cc1" {{.*}}
//  CHECK-OFFLOAD-NEXT-SAME: "-fsycl-is-host"
//  CHECK-OFFLOAD-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]"

/// Check "-aux-target-cpu" with "-aux-target-feature" and "-target-cpu" with "-target-feature" are passed
/// when compiling for SYCL offload device and host codes:
//  RUN:  %clang -fsycl -mavx -c %s -### -o %t.o 2>&1 | FileCheck -check-prefix=OFFLOAD-AVX %s
//  OFFLOAD-AVX: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  OFFLOAD-AVX-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]" "-aux-target-feature" "+avx"
//  OFFLOAD-AVX-NEXT: clang{{.*}} "-cc1" {{.*}}
//  OFFLOAD-AVX-NEXT-SAME: "-fsycl-is-host"
//  OFFLOAD-AVX-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]" "-target-feature" "+avx"

/// Check that the needed -fsycl -fsycl-is-device and -fsycl-is-host options
/// are passed to all of the needed steps with and without preprocessing
// RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl -E %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl" "-fsycl-is-device"
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl" "-fsycl-is-host"
