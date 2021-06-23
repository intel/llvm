// Tests the abilities involved with using an external host compiler
// REQUIRES: clang-driver

/// enabling with -fsycl-host-compiler
// RUN: %clangxx -fsycl-use-footer -fsycl -fsycl-host-compiler=/some/dir/g++ %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER %s
// HOST_COMPILER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"
// HOST_COMPILER: g++{{.*}} "-E" "-include" "[[INTHEADER]]" "-I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-o" "[[TMPII:.+\.ii]]"
// HOST_COMPILER: g++{{.*}} "-c" "-I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-o" "[[HOSTOBJ:.+\.o]]"{{.*}}
// HOST_COMPILER: ld{{.*}} "[[HOSTOBJ]]"

// RUN: %clang_cl -fsycl-use-footer -fsycl -fsycl-host-compiler=/some/dir/cl %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_CL %s
// HOST_COMPILER_CL: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"
// HOST_COMPILER_CL: cl{{.*}} "-P" "-Fi[[TMPII:.+\.ii]]" "-FI" "[[INTHEADER]]"{{.*}} "-I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// HOST_COMPILER_CL: cl{{.*}} "-c" "-Fo[[HOSTOBJ:.+\.obj]]"{{.*}} "-I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}}
// HOST_COMPILER_CL: link{{.*}} "[[HOSTOBJ]]"

/// check for additional host options
// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-DFOO -DBAR" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS %s
// HOST_OPTIONS: g++{{.*}} "-o" "[[HOSTOBJ:.+\.o]]"{{.*}} "-DFOO" "-DBAR"

// RUN: %clang_cl -fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options="/DFOO /DBAR /O2" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS_CL %s
// HOST_OPTIONS_CL: cl{{.*}} "-Fo[[HOSTOBJ:.+\.obj]]"{{.*}} "/DFOO" "/DBAR" "/O2"

/// preprocessing
// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -E %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_PREPROCESS %s
// HOST_PREPROCESS: g++{{.*}} "-E"{{.*}} "-o" "[[PPOUT:.+\.ii]]"
// HOST_PREPROCESS: clang-offload-bundler{{.*}} "-inputs={{.*}}.ii,[[PPOUT]]"

// RUN: %clang_cl -fsycl -fsycl-host-compiler=cl -E %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_PREPROCESS_CL %s
// HOST_PREPROCESS_CL: cl{{.*}} "-P"{{.*}} "-Fi[[PPOUT:.+\.ii]]"
// HOST_PREPROCESS_CL: clang-offload-bundler{{.*}} "-inputs={{.*}}.ii,[[PPOUT]]"

/// obj output
// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT %s
// HOST_OBJECT: g++{{.*}} "-c"{{.*}} "-o" "[[OBJOUT:.+\.o]]"
// HOST_OBJECT: clang-offload-bundler{{.*}} "-inputs={{.*}}.bc,[[OBJOUT]]"

// RUN: %clang_cl -fsycl -fsycl-host-compiler=cl -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT_CL %s
// HOST_OBJECT_CL: cl{{.*}} "-c"{{.*}} "-Fo[[OBJOUT:.+\.obj]]"
// HOST_OBJECT_CL: clang-offload-bundler{{.*}} "-inputs={{.*}}.bc,[[OBJOUT]]"

/// assembly output
// RUN: %clangxx -fsycl -fsycl-host-compiler=g++ -S %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_ASSEMBLY %s
// HOST_ASSEMBLY: g++{{.*}} "-S"{{.*}} "-o" "[[ASMOUT:.+\.s]]"
// HOST_ASSEMBLY: clang-offload-bundler{{.*}} "-inputs={{.*}}.bc,[[ASMOUT]]"

// RUN: %clangxx -fsycl -fsycl-host-compiler=cl -S %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_ASSEMBLY_CL %s
// HOST_ASSEMBLY_CL: cl{{.*}} "-c"{{.*}} "-Fa[[ASMOUT:.+\.s]]" "-Fo{{.*}}.obj"
// HOST_ASSEMBLY_CL: clang-offload-bundler{{.*}} "-inputs={{.*}}.bc,[[ASMOUT]]"

/// missing argument error -fsycl-host-compiler=
// RUN: %clangxx -fsycl -fsycl-host-compiler= -c -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_NOARG %s
// HOST_COMPILER_NOARG: missing argument to '-fsycl-host-compiler='
