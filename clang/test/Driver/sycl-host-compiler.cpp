// Tests the abilities involved with using an external host compiler
// with the new offload model.

/// Enabling with -fsycl-host-compiler
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-host-compiler=/some/dir/g++ %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER %s
// HOST_COMPILER: clang{{.*}} "-fsycl-is-device"
// HOST_COMPILER-SAME: "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"
// HOST_COMPILER-SAME: "-o" "[[DEVICEBC:.+\.bc]]"
// HOST_COMPILER: append-file{{.*}} "--append=[[INTFOOTER]]"
// HOST_COMPILER-SAME: "--output=[[APPENDFILESRC:.+\.cpp]]" "--use-include"
// HOST_COMPILER: g++{{.*}} "[[APPENDFILESRC]]"
// HOST_COMPILER-SAME: "-c" "-include" "[[INTHEADER]]"
// HOST_COMPILER-SAME: "-iquote"
// HOST_COMPILER-SAME: "-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// HOST_COMPILER-SAME: "-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// HOST_COMPILER-SAME: "-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"
// HOST_COMPILER-SAME: "-o" "[[HOSTOBJ:.+\.o]]"
// HOST_COMPILER: clang-offload-bundler{{.*}} "-output=[[BUNDLEOBJ:.+\.o]]" "-input=[[DEVICEBC]]" "-input=[[HOSTOBJ]]"
// HOST_COMPILER: clang-linker-wrapper{{.*}} "[[BUNDLEOBJ]]"

// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-host-compiler=/some/dir/cl %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_CL %s
// HOST_COMPILER_CL: clang{{.*}} "-fsycl-is-device"
// HOST_COMPILER_CL-SAME: "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"
// HOST_COMPILER_CL-SAME: "-o" "[[DEVICEBC:.+\.bc]]"
// HOST_COMPILER_CL: append-file{{.*}} "--append=[[INTFOOTER]]"
// HOST_COMPILER_CL-SAME: "--output=[[APPENDFILESRC:.+\.cpp]]" "--use-include"
// HOST_COMPILER_CL: cl{{.*}} "[[APPENDFILESRC]]"
// HOST_COMPILER_CL-SAME: "-c" "-Fo[[HOSTOBJ:.+\.obj]]" "-FI" "[[INTHEADER]]"
// HOST_COMPILER_CL-SAME: "/external:W0"
// HOST_COMPILER_CL-SAME: "/external:I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// HOST_COMPILER_CL-SAME: "/external:I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"
// HOST_COMPILER_CL-SAME: "/external:I" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"
// HOST_COMPILER_CL: clang-offload-bundler{{.*}} "-output=[[BUNDLEOBJ:.+\.obj]]" "-input=[[DEVICEBC]]" "-input=[[HOSTOBJ]]"
// HOST_COMPILER_CL: clang-linker-wrapper{{.*}} "[[BUNDLEOBJ]]"

/// Check for additional host options.
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-DFOO -DBAR" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS %s
// HOST_OPTIONS: g++{{.*}} "-o" "[[HOSTOBJ:.+\.o]]"{{.*}} "-DFOO" "-DBAR"

// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-host-compiler=cl -fsycl-host-compiler-options="/DFOO /DBAR /O2" %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OPTIONS_CL %s
// HOST_OPTIONS_CL: cl{{.*}} "-Fo[[HOSTOBJ:.+\.obj]]"{{.*}} "/DFOO" "/DBAR" "/O2"

/// Object output check.
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-host-compiler=g++ -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT %s
// HOST_OBJECT: g++{{.*}} "-c"{{.*}} "-o" "[[OBJOUT:.+\.o]]"
// HOST_OBJECT: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[OBJOUT]]"

// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-host-compiler=cl -c %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_OBJECT_CL %s
// HOST_OBJECT_CL: cl{{.*}} "-c"{{.*}} "-Fo[[OBJOUT:.+\.obj]]"
// HOST_OBJECT_CL: clang-offload-bundler{{.*}} "-input={{.*}}.bc" "-input=[[OBJOUT]]"

/// Missing argument error -fsycl-host-compiler=.
// RUN: not %clangxx -fsycl --offload-new-driver -fsycl-host-compiler= -c -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_NOARG %s
// HOST_COMPILER_NOARG: missing argument to '-fsycl-host-compiler='

/// Error for -fsycl-host-compiler and -fsycl-unnamed-lambda combination.
// RUN: not %clangxx -fsycl --offload-new-driver -fsycl-host-compiler=g++ -fsycl-unnamed-lambda -c -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=HOST_COMPILER_AND_UNNAMED_LAMBDA %s
// HOST_COMPILER_AND_UNNAMED_LAMBDA: error: cannot specify '-fsycl-unnamed-lambda' along with '-fsycl-host-compiler'

// -fsycl-host-compiler implies -fno-sycl-unnamed-lambda.
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-host-compiler=g++ -c -### %s 2>&1 \
// RUN:  | FileCheck -check-prefix=IMPLY-NO-SYCL-UNNAMED-LAMBDA %s
// IMPLY-NO-SYCL-UNNAMED-LAMBDA: clang{{.*}} "-fno-sycl-unnamed-lambda"

// Zc:__cplusplus, Zc:__cplusplus- check.
// RUN: %clang_cl -### -fsycl-host-compiler=cl -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-ZC-CPLUSPLUS %s
// RUN: %clang_cl -### -fsycl-host-compiler=cl -fsycl --offload-new-driver -fsycl-host-compiler-options=/Zc:__cplusplus- %s 2>&1 | FileCheck -check-prefix=CHECK-ZC-CPLUSPLUS-MINUS %s
// RUN: %clang_cl -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ZC-CPLUSPLUS %s
// RUN: %clang_cl -### -fsycl-host-compiler=g++ -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ZC-CPLUSPLUS %s
// CHECK-ZC-CPLUSPLUS: "/Zc:__cplusplus"
// CHECK-ZC-CPLUSPLUS-MINUS: "/Zc:__cplusplus-"
// CHECK-NO-ZC-CPLUSPLUS-NOT: "/Zc:__cplusplus"

/// -fsycl-host-compiler -save-temps behavior
// RUN: %clangxx -### -fsycl-host-compiler=g++ -fsycl --offload-new-driver \
// RUN:          -save-temps -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SAVE_TEMPS %s
// CHECK_SAVE_TEMPS-NOT: error: unsupported output type when using external host compiler
// CHECK_SAVE_TEMPS: clang{{.*}} "-fsycl-is-device"
// CHECK_SAVE_TEMPS-SAME: "-E" {{.*}} "-o" "[[PREPROC_OUT:.+sycl-spir64-unknown-unknown.ii]]"
// CHECK_SAVE_TEMPS-NEXT: clang{{.*}} "-fsycl-is-device"
// CHECK_SAVE_TEMPS-SAME: "-emit-llvm-bc"{{.*}} "-o" "[[DEVICE_BC1:.+\.bc]]"{{.*}} "[[PREPROC_OUT]]"
// CHECK_SAVE_TEMPS-NEXT: clang{{.*}} "-fsycl-is-device"
// CHECK_SAVE_TEMPS-SAME: "-emit-llvm-bc"{{.*}} "-o" "[[DEVICE_BC2:.+\.bc]]"{{.*}} "[[DEVICE_BC1]]"
// CHECK_SAVE_TEMPS-NEXT: append-file{{.*}} "--output=[[APPEND_CPP:.+\.cpp]]
// CHECK_SAVE_TEMPS-NEXT: g++{{.*}} "[[APPEND_CPP]]" "-c"
// CHECK_SAVE_TEMPS-SAME: "-o" "[[HOST_OBJ:.+\.o]]"
// CHECK_SAVE_TEMPS-NEXT: clang-offload-bundler{{.*}} "-input=[[DEVICE_BC2]]" "-input=[[HOST_OBJ]]"

/// Test to verify binary from PATH is used
// RUN: rm -rf %t && mkdir -p %t/test_path
// RUN: touch %t/test_path/clang++ && chmod +x %t/test_path/clang++
// RUN: env "PATH=%t/test_path%{pathsep}%PATH%" \
// RUN: %clangxx -### -fsycl -fsycl-host-compiler=clang++ \
// RUN:   -fsycl-host-compiler-options=-DDUMMY_OPT --offload-new-driver \
// RUN:   %s 2>&1 \
// RUN: | FileCheck -check-prefix=PATH_CHECK %s
// PATH_CHECK: {{(/|\\\\)}}test_path{{(/|\\\\)}}clang++{{.*}} "-DDUMMY_OPT"
