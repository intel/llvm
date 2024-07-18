// Tests the abilities involved with using an external host compiler

/// enabling with -fsycl-host-compiler
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=/some/dir/g++ %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER %s
// HOST_COMPILER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"
// HOST_COMPILER: g++{{.*}} "-c" "-include" "[[INTHEADER]]" "-iquote" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-o" "[[HOSTOBJ:.+\.o]]"{{.*}}
// HOST_COMPILER: ld{{.*}} "[[HOSTOBJ]]"

// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-host-compiler=/some/dir/cl %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_CL %s
// HOST_COMPILER_CL: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer={{.*}}"
// HOST_COMPILER_CL: cl{{.*}} "-c" "-Fo[[HOSTOBJ:.+\.obj]]" "-FI" "[[INTHEADER]]"{{.*}} "-I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}}
// HOST_COMPILER_CL: link{{.*}} "[[HOSTOBJ]]"

/// check for additional host options
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-DFOO -DBAR" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS %s
// HOST_OPTIONS: g++{{.*}} "-o" "[[HOSTOBJ:.+\.o]]"{{.*}} "-DFOO" "-DBAR"

// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-host-compiler=cl -fsycl-host-compiler-options="/DFOO /DBAR /O2" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS_CL %s
// HOST_OPTIONS_CL: cl{{.*}} "-Fo[[HOSTOBJ:.+\.obj]]"{{.*}} "/DFOO" "/DBAR" "/O2"

/// preprocessing
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ -E %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_PREPROCESS %s
// HOST_PREPROCESS: append-file{{.*}} "--output=[[APPEND:.+\.cpp]]"
// HOST_PREPROCESS: g++{{.*}} "[[APPEND]]"{{.*}} "-E"{{.*}} "-o" "[[PPOUT2:.+\.ii]]"
// HOST_PREPROCESS: clang-offload-bundler{{.*}} "-input={{.*}}.ii" "-input=[[PPOUT2]]"

// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-host-compiler=cl -E %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_PREPROCESS_CL %s
// HOST_PREPROCESS_CL: append-file{{.*}} "--output=[[APPEND:.+\.cpp]]"
// HOST_PREPROCESS_CL: cl{{.*}} "[[APPEND]]"{{.*}} "-P"{{.*}} "-Fi[[PPOUT2:.+\.ii]]"
// HOST_PREPROCESS_CL: clang-offload-bundler{{.*}} "-input={{.*}}.ii" "-input=[[PPOUT2]]"

/// obj output
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT %s
// HOST_OBJECT: g++{{.*}} "-c"{{.*}} "-o" "[[OBJOUT:.+\.o]]"
// HOST_OBJECT: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[OBJOUT]]"

// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-host-compiler=cl -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT_CL %s
// HOST_OBJECT_CL: cl{{.*}} "-c"{{.*}} "-Fo[[OBJOUT:.+\.obj]]"
// HOST_OBJECT_CL: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[OBJOUT]]"

/// assembly output
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ -S %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_ASSEMBLY %s
// HOST_ASSEMBLY: g++{{.*}} "-S"{{.*}} "-o" "[[ASMOUT:.+\.s]]"
// HOST_ASSEMBLY: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[ASMOUT]]"

// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=cl -S %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_ASSEMBLY_CL %s
// HOST_ASSEMBLY_CL: cl{{.*}} "-c"{{.*}} "-Fa[[ASMOUT:.+\.s]]" "-Fo{{.*}}.obj"
// HOST_ASSEMBLY_CL: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[ASMOUT]]"

/// missing argument error -fsycl-host-compiler=
// RUN: not %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler= -c -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_NOARG %s
// HOST_COMPILER_NOARG: missing argument to '-fsycl-host-compiler='

/// Warning should not be emitted when using -fsycl-host-compiler when linking
// RUN: touch %t.o
// RUN: %clangxx -fsycl --no-offload-new-driver -fsycl-host-compiler=g++ %t.o -### 2>&1 \
// RUN:  | FileCheck -check-prefix=WARNING_HOST_COMPILER %s
// WARNING_HOST_COMPILER-NOT: warning: argument unused during compilation: '-fsycl-host-compiler=g++' [-Wunused-command-line-argument]

// Zc:__cplusplus, Zc:__cplusplus- check
// RUN: %clang_cl -### -fsycl-host-compiler=cl -fsycl --no-offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-ZC-CPLUSPLUS %s
// RUN: %clang_cl -### -fsycl-host-compiler=cl -fsycl --no-offload-new-driver -fsycl-host-compiler-options=/Zc:__cplusplus- %s 2>&1 | FileCheck -check-prefix=CHECK-ZC-CPLUSPLUS-MINUS %s
// RUN: %clang_cl -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ZC-CPLUSPLUS %s
// RUN: %clang_cl -### -fsycl-host-compiler=g++ -fsycl --no-offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ZC-CPLUSPLUS %s
// CHECK-ZC-CPLUSPLUS: "/Zc:__cplusplus"
// CHECK-ZC-CPLUSPLUS-MINUS: "/Zc:__cplusplus-"
// CHECK-NO-ZC-CPLUSPLUS-NOT: "/Zc:__cplusplus"
