// RUN: %clang -c -fsycl -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -fsycl -foffload-fp32-prec-div -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -fsycl -foffload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt \
// RUN: -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: -### %s 2>&1 | FileCheck %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_SQRT %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1	\
// RUN: | FileCheck --check-prefix=NO_PREC_DIV_SQRT %s

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-DIV,NO_PREC_DIV_FP_ACC_HIGH

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-DIV,NO_PREC_DIV_FP_ACC_HIGH

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-div -ffp-accuracy=high:fdiv \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-DIV-ONLY,FP_ACC_HIGH_DIV

// RUN: %clang -c -fsycl -ffp-accuracy=high:fdiv \
// RUN: -fno-math-errno -fno-offload-fp32-prec-div  -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-DIV-ONLY,NO_PREC_DIV

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -ffp-accuracy=high \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-SQRT,NO_PREC_SQRT_FP_ACC_HIGH

// RUN: %clang -c -fsycl -fno-offload-fp32-prec-sqrt -ffp-accuracy=high:sqrt \
// RUN: -fno-math-errno -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-SQRT-ONLY,FP_ACC_HIGH_SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=high:sqrt \
// RUN: -fno-math-errno -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-SQRT-ONLY,NO_PREC_SQRT

// RUN: %clang -c -fsycl -ffp-accuracy=high -fno-math-errno \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-HIGH-SQRT,NO_PREC_SQRT_FP_ACC_HIGH

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-LOW-DIV,NO_PREC_DIV_FP_ACC_LOW

// RUN: %clang -c -fsycl -ffp-accuracy=low -fno-math-errno \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=WARN-LOW-SQRT,NO_PREC_SQRT_FP_ACC_LOW

// RUN: %clang -c -fsycl -ffp-model=fast  -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FAST %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-div -ffp-model=fast -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FAST %s

// RUN: %clang -c -fsycl -foffload-fp32-prec-sqrt -ffp-model=fast -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FAST %s

// RUN: %clang -c -fsycl -ffp-model=fast -foffload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_SQRT %s

// RUN: %clang -c -fsycl -ffp-model=fast -foffload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV %s

// RUN: %clang_cl -c -fsycl -foffload-fp32-prec-div -### %s 2>&1 | FileCheck %s
// RUN: %clang_cl -c -fsycl -foffload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang_cl -c -fsycl -foffload-fp32-prec-div \
// RUN: -foffload-fp32-prec-sqrt -### %s 2>&1 | FileCheck %s

// RUN: %clang_cl -c -fsycl -foffload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: -### %s 2>&1 | FileCheck %s

// RUN: %clang_cl -c -fsycl -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_DIV %s

// RUN: %clang_cl -c -fsycl -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_PREC_SQRT %s

// RUN: %clang_cl -c -fsycl -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FAST %s

// RUN: %clang_cl -c -fsycl -fno-offload-fp32-prec-sqrt \
// RUN: -fno-offload-fp32-prec-div -### %s 2>&1 \
// RUN: | FileCheck --check-prefix=FAST %s

// WARN-HIGH-DIV: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-offload-fp32-prec-div'

// WARN-HIGH-DIV-ONLY: floating point accuracy control 'high:fdiv' conflicts with explicit target precision option '-fno-offload-fp32-prec-div'

// WARN-HIGH-SQRT: floating point accuracy control 'high' conflicts with explicit target precision option '-fno-offload-fp32-prec-sqrt'

// WARN-HIGH-SQRT-ONLY: floating point accuracy control 'high:sqrt' conflicts with explicit target precision option '-fno-offload-fp32-prec-sqrt'

// WARN-LOW-DIV: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-offload-fp32-prec-div'

// WARN-LOW-SQRT: floating point accuracy control 'low' conflicts with explicit target precision option '-fno-offload-fp32-prec-sqrt'


// CHECK: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}}
// CHECK-NOT: "-foffload-fp32-prec-div"
// CHECK-NOT: "-foffload-fp32-prec-sqrt"

// NO_PREC_DIV: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div"

// NO_PREC_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-sqrt"

// NO_PREC_DIV_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-fno-offload-fp32-prec-sqrt"

// FAST: "-triple" "spir64{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-fno-offload-fp32-prec-sqrt"

// FP_ACC_HIGH_DIV: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-ffp-builtin-accuracy=high:fdiv"

// FP_ACC_HIGH_SQRT: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-ffp-builtin-accuracy=high:sqrt"

// NO_PREC_DIV_FP_ACC_HIGH: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-ffp-builtin-accuracy=high"

// NO_PREC_SQRT_FP_ACC_HIGH: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}}  "-fno-offload-fp32-prec-sqrt" "-ffp-builtin-accuracy=high"

// NO_PREC_DIV_FP_ACC_LOW: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-div" "-ffp-builtin-accuracy=low"

// NO_PREC_SQRT_FP_ACC_LOW: "-triple" "spir64{{.*}}" "-fsycl-is-device"{{.*}} "-fno-offload-fp32-prec-sqrt" "-ffp-builtin-accuracy=low"
