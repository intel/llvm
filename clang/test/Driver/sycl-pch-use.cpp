// This test checks that a PCH(Precompiled Header) file is 
// used while performing host and device compilations in
// -fsycl mode.

// RUN: echo "" > %t.h
// RUN: %clang -c -x c++-header %t.h

// Linux
// RUN: %clang -fsycl -c -include-pch %t.h.pch %s -### 2>&1 | FileCheck -check-prefix=LX_USE %s
// LX_USE: clang-offload-bundler{{.*}} "-type=pch"
// LX_USE-SAME: "-targets=host-x86_64{{.*}},sycl-spir64{{.*}}" "-input=[[MAINPCHFILE:.+\.pch]]" "-output=[[PCHFILE1:.+\.pch]]" "-output=[[PCHFILE2:.+\.pch]]" "-unbundle"
// LX_USE: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// LX_USE-SAME: "-include-pch" "[[PCHFILE2]]"{{.*}}
// LX_USE: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// LX_USE-SAME: "-include-pch" "[[PCHFILE1]]"{{.*}}

// RUN: %clang -fsycl -c -include-pch %t.h.pch -fsycl-targets=spir64,spir64_gen %s -### 2>&1 | FileCheck -check-prefix=LX_USE_TARGETS %s
// LX_USE_TARGETS: clang-offload-bundler{{.*}} "-type=pch"
// LX_USE_TARGETS-SAME: "-targets=host-x86_64{{.*}},sycl-spir64{{.*}},sycl-spir64_gen{{.*}}" "-input=[[MAINPCHFILE:.+\.pch]]" "-output=[[PCHFILE3:.+\.pch]]" "-output=[[PCHFILE4:.+\.pch]]" "-output=[[PCHFILE5:.+\.pch]]" "-unbundle"
// LX_USE_TARGETS: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// LX_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE4]]"{{.*}}
// LX_USE_TARGETS: clang{{.*}} "-triple" "spir64_gen{{.*}}"{{.*}} "-fsycl-is-device"
// LX_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE5]]"{{.*}}
// LX_USE_TARGETS: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// LX_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE3]]"{{.*}}

// Windows
// RUN: %clang_cl -fsycl -c /Yu%t.h %s -### 2>&1 | FileCheck -check-prefix=WS_USE %s
// WS_USE: clang-offload-bundler{{.*}} "-type=pch"
// WS_USE-SAME: "-targets=host-x86_64{{.*}},sycl-spir64{{.*}}" "-input=[[MAINPCHFILE:.+\.pch]]" "-output=[[PCHFILE1:.+\.pch]]" "-output=[[PCHFILE2:.+\.pch]]" "-unbundle"
// WS_USE: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// WS_USE-SAME: "-include-pch" "[[PCHFILE2]]"{{.*}}
// WS_USE: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// WS_USE-SAME: "-include-pch" "[[PCHFILE1]]"{{.*}}

/ Windows
// RUN: %clang_cl -fsycl -fsycl-targets=spir64,spir64_gen -c /Yu%t.h %s -### 2>&1 | FileCheck -check-prefix=WS_USE_TARGETS %s
// WS_USE_TARGETS: clang-offload-bundler{{.*}} "-type=pch"
// WS_USE_TARGETS-SAME: "-targets=host-x86_64{{.*}},sycl-spir64{{.*}},sycl-spir64_gen{{.*}}" "-input=[[MAINPCHFILE:.+\.pch]]" "-output=[[PCHFILE3:.+\.pch]]" "-output=[[PCHFILE4:.+\.pch]]" "-output=[[PCHFILE5:.+\.pch]]" "-unbundle"
// WS_USE_TARGETS: clang{{.*}} "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"
// WS_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE4]]"{{.*}}
// WS_USE_TARGETS: clang{{.*}} "-triple" "spir64_gen{{.*}}"{{.*}} "-fsycl-is-device"
// WS_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE5]]"{{.*}}
// WS_USE_TARGETS: clang{{.*}} "-triple" "x86_64{{.*}}"{{.*}} "-fsycl-is-host"
// WS_USE_TARGETS-SAME: "-include-pch" "[[PCHFILE3]]"{{.*}}
