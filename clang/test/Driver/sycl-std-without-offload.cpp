/// Check error when -sycl-std is used without -fsycl option.
// RUN:   %clang -### -sycl-std=2017  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FSYCL-SYCL-STD %s
// CHK-NO-FSYCL-SYCL-STD: error: '-sycl-std' must be used in conjunction with '-fsycl' to enable offloading
