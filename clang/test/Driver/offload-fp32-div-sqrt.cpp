// RUN: %clang -c -fsycl -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-div -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-sqrt -foffload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_SQRT %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt -### %s 2>&1\
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -fno-offload-fp32-prec-div -### %s 2>&1\
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-DIV

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-DIV

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno -fno-offload-fp32-prec-div \
// RUN: -### %s 2>&1  | FileCheck %s --check-prefix=WARN-LOW-DIV

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-LOW-SQRT

// CHECK: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-foffload-fp32-prec-div" "-foffload-fp32-prec-sqrt"
// CHECK-NOT: "-triple{{.*}}" "-fsycl-is-host"{{.*}} "-foffload-fp32-prec-div" "-foffload-fp32-prec-sqrt"
// NO_PREC_DIV: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-foffload-fp32-prec-sqrt"
// NO_PREC_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-foffload-fp32-prec-div" "-fno-offload-fp32-prec-sqrt"
// NO_PREC_DIV_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-fno-offload-fp32-prec-sqrt"
// RUN: %clang -c -fsycl -ffp-model=fast  -### %s 2>&1 | FileCheck --check-prefix=FAST %s

// WARN-HIGH-DIV: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-offload-fp32-prec-div'
// WARN-HIGH-SQRT: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-offload-fp32-prec-sqrt'
// WARN-LOW-DIV: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-offload-fp32-prec-div'
// WARN-LOW-SQRT: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-offload-fp32-prec-sqrt'
// FAST: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-fno-offload-fp32-prec-sqrt"

