// This test checks that a PCH(Precompiled Header) file is
// created while performing host and device compilations in
// -fsycl mode.

// RUN: echo "// Header file" > %t.h

// Linux
// RUN: %clang -fsycl -x c++-header -c %t.h %s -### 2>&1 | FileCheck -check-prefix=LX_CREATE %s
// LX_CREATE: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// LX_CREATE-SAME: "-fsycl-int-header=[[HEADER1:.+\.h]]"{{.*}} "-fsycl-int-footer=[[FOOTER1:.+\.h]]"{{.*}} "-sycl-std=2020"
// LX_CREATE-SAME: "-o" "[[PCHFILE1:.+pch]]"
// LX_CREATE: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// LX_CREATE-SAME: "-include-internal-header" "[[HEADER1]]"
// LX_CREATE-SAME: "-include-internal-footer" "[[FOOTER1]]"
// LX_CREATE-SAME: "-o" "[[PCHFILE2:.+pch]]"
// LX_CREATE: clang-offload-bundler{{.*}} "-type=pch"
// LX_CREATE: "-targets=sycl-spir64{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]"

// RUN: %clang -fsycl -x c++-header -fsycl-targets=spir64,spir64_gen -c %t.h %s -### 2>&1 | FileCheck -check-prefix=LX_CREATE_TARGETS %s
// LX_CREATE_TARGETS: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// LX_CREATE_TARGETS-SAME: "-fsycl-int-header=[[HEADER1:.+\.h]]"{{.*}} "-fsycl-int-footer=[[FOOTER1:.+\.h]]"{{.*}} "-sycl-std=2020"
// LX_CREATE_TARGETS-SAME: "-o" "[[PCHFILE1:.+pch]]"
// LX_CREATE_TARGETS: clang{{.*}} "-triple" "spir64_gen{{.*}}"{{.*}} "-fsycl-is-device"
// LX_CREATE_TARGETS-SAME: "-fsycl-int-header=[[HEADER1]]"{{.*}} "-fsycl-int-footer=[[FOOTER1]]"{{.*}} "-sycl-std=2020"
// LX_CREATE_TARGETS-SAME: "-o" "[[PCHFILE2:.+pch]]"
// LX_CREATE_TARGETS: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// LX_CREATE_TARGETS-SAME: "-include-internal-header" "[[HEADER1]]"
// LX_CREATE_TARGETS-SAME: "-include-internal-footer" "[[FOOTER1]]"
// LX_CREATE_TARGETS-SAME: "-o" "[[PCHFILE3:.+pch]]"
// LX_CREATE_TARGETS: clang-offload-bundler{{.*}} "-type=pch"
// LX_CREATE_TARGETS: "-targets=sycl-spir64{{.*}},sycl-spir64_gen{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]" "-input=[[PCHFILE3]]"

// RUN: %clang -fsycl -x c++-header -fno-sycl-use-header -fno-sycl-use-footer -c %t.h %s -### 2>&1 | FileCheck -check-prefix=LX_CREATE_NOHF %s
// LX_CREATE_NOHF: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// LX_CREATE_NOHF-NOT: "-fsycl-int-header={{.*}}"{{.*}} "-fsycl-int-footer={{.*}}"{{.*}}
// LX_CREATE_NOHF-SAME: "-o" "[[PCHFILE1:.+pch]]"
// LX_CREATE_NOHF: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// LX_CREATE_NOHF-NOT: "-include-internal-header"
// LX_CREATE_NOHF-NOT: "-include-internal-footer"
// LX_CREATE_NOHF-SAME: "-o" "[[PCHFILE2:.+pch]]"
// LX_CREATE_NOHF: clang-offload-bundler{{.*}} "-type=pch"
// LX_CREATE_NOHF: "-targets=sycl-spir64{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]"

// Windows
// RUN: %clang_cl -fsycl -x c++-header %t.h %s -### 2>&1 | FileCheck -check-prefix=WS_CREATE %s
// WS_CREATE: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// WS_CREATE-SAME: "-fsycl-int-header=[[HEADER1:.+\.h]]"{{.*}} "-fsycl-int-footer=[[FOOTER1:.+\.h]]"{{.*}} "-sycl-std=2020"
// WS_CREATE-SAME: "-o" "[[PCHFILE1:.+pch]]"
// WS_CREATE: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// WS_CREATE-SAME: "-include-internal-header" "[[HEADER1]]"
// WS_CREATE-SAME: "-include-internal-footer" "[[FOOTER1]]"
// WS_CREATE-SAME: "-o" "[[PCHFILE2:.+pch]]"
// WS_CREATE: clang-offload-bundler{{.*}} "-type=pch"
// WS_CREATE: "-targets=sycl-spir64{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]"

// RUN: %clang_cl -fsycl -x c++-header -fsycl-targets=spir64,spir64_gen -c %t.h %s -### 2>&1 | FileCheck -check-prefix=WS_CREATE_TARGETS %s
// WS_CREATE_TARGETS: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// WS_CREATE_TARGETS-SAME: "-fsycl-int-header=[[HEADER1:.+\.h]]"{{.*}} "-fsycl-int-footer=[[FOOTER1:.+\.h]]"{{.*}} "-sycl-std=2020"
// WS_CREATE_TARGETS-SAME: "-o" "[[PCHFILE1:.+pch]]"
// WS_CREATE_TARGETS: clang{{.*}} "-triple" "spir64_gen{{.*}}"{{.*}} "-fsycl-is-device"
// WS_CREATE_TARGETS-SAME: "-fsycl-int-header=[[HEADER1]]"{{.*}} "-fsycl-int-footer=[[FOOTER1]]"{{.*}} "-sycl-std=2020"
// WS_CREATE_TARGETS-SAME: "-o" "[[PCHFILE2:.+pch]]"
// WS_CREATE_TARGETS: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// WS_CREATE_TARGETS-SAME: "-include-internal-header" "[[HEADER1]]"
// WS_CREATE_TARGETS-SAME: "-include-internal-footer" "[[FOOTER1]]"
// WS_CREATE_TARGETS-SAME: "-o" "[[PCHFILE3:.+pch]]"
// WS_CREATE_TARGETS: clang-offload-bundler{{.*}} "-type=pch"
// WS_CREATE_TARGETS: "-targets=sycl-spir64{{.*}},sycl-spir64_gen{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]" "-input=[[PCHFILE3]]"

// RUN: %clang_cl -fsycl -x c++-header -fno-sycl-use-header -fno-sycl-use-footer -c %t.h %s -### 2>&1 | FileCheck -check-prefix=WS_CREATE_NOHF %s
// WS_CREATE_NOHF: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// WS_CREATE_NOHF-NOT: "-fsycl-int-header={{.*}}"{{.*}} "-fsycl-int-footer={{.*}}"{{.*}}
// WS_CREATE_NOHF-SAME: "-o" "[[PCHFILE1:.+pch]]"
// WS_CREATE_NOHF: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// WS_CREATE_NOHF-NOT: "-include-internal-header"
// WS_CREATE_NOHF-NOT: "-include-internal-footer"
// WS_CREATE_NOHF-SAME: "-o" "[[PCHFILE2:.+pch]]"
// WS_CREATE_NOHF: clang-offload-bundler{{.*}} "-type=pch"
// WS_CREATE_NOHF: "-targets=sycl-spir64{{.*}},host-x86_64{{.*}}" "-output={{.*}}.pch" "-input=[[PCHFILE1]]" "-input=[[PCHFILE2]]"
