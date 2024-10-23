// RUN: %clang -c -fsycl -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -ftarget-prec-div -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -ftarget-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -ftarget-prec-div -ftarget-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s

// RUN: %clang -c -fsycl -ftarget-prec-sqrt -ftarget-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s

// RUN: %clang -c -fsycl -fno-target-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV %s

// RUN: %clang -c -fsycl -fno-target-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_SQRT %s

// RUN: %clang -c -fsycl -fno-target-prec-div -fno-target-prec-sqrt -### %s 2>&1\
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -fno-target-prec-sqrt -fno-target-prec-div -### %s 2>&1\
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-target-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-DIV

// RUN: %clang -c -fsycl -fno-target-prec-div -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-DIV

// RUN: %clang -c -fsycl -fno-target-prec-sqrt -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-target-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-HIGH-SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno -fno-target-prec-div \
// RUN: -### %s 2>&1  | FileCheck %s --check-prefix=WARN-LOW-DIV

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno \
// RUN: -fno-target-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=WARN-LOW-SQRT

// CHECK: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-ftarget-prec-div" "-ftarget-prec-sqrt"
// CHECK-NOT: "-triple{{.*}}" "-fsycl-is-host"{{.*}} "-ftarget-prec-div" "-ftarget-prec-sqrt"
// NO_PREC_DIV: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-target-prec-div" "-ftarget-prec-sqrt"
// NO_PREC_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-ftarget-prec-div" "-fno-target-prec-sqrt"
// NO_PREC_DIV_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-target-prec-div" "-fno-target-prec-sqrt"
// RUN: %clang -c -fsycl -ffp-model=fast  -### %s 2>&1 | FileCheck --check-prefix=FAST %s

// WARN-HIGH-DIV: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-target-prec-div'
// WARN-HIGH-SQRT: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-target-prec-sqrt'
// WARN-LOW-DIV: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-target-prec-div'
// WARN-LOW-SQRT: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-target-prec-sqrt'
// FAST: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-target-prec-div" "-fno-target-prec-sqrt"

