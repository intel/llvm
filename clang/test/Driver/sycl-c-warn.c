// Emit warning for treating 'c' input as 'c++' when -fsycl is used
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang_cl -### -fsycl %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang -### -fsycl --offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang_cl -### -fsycl --offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang -### -fsycl --no-offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver  %s 2>&1 | FileCheck -check-prefix FSYCL-CHECK %s
// FSYCL-CHECK: warning: treating 'c' input as 'c++' when -fsycl is used [-Wexpected-file-type]

