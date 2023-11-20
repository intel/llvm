// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -S -emit-mlir -o - %s -fsycl-device-only -w -ffp-model=precise | FileCheck %s --check-prefixes="CHECK,CHECK-PRESTRICT,CHECK-PRECISE"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -S -emit-mlir -o - %s -fsycl-device-only -w -ffp-model=strict | FileCheck %s --check-prefixes="CHECK,CHECK-PRESTRICT,CHECK-STRICT"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -S -emit-mlir -o - %s -fsycl-device-only -w -ffp-model=fast | FileCheck %s --check-prefixes="CHECK,CHECK-FAST"
// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -S -emit-mlir -o - %s -fsycl-device-only -w -fno-honor-nans -fno-honor-infinities | FileCheck %s --check-prefixes="CHECK,CHECK-NOHONOR"

#include <sycl/sycl.hpp>

// COM: Check fast math flags are propagated.

template <typename T> SYCL_EXTERNAL T add(T a, T b) { return a + b; }
template <typename T> SYCL_EXTERNAL T sub(T a, T b) { return a - b; }
template <typename T> SYCL_EXTERNAL T mul(T a, T b) { return a * b; }
template <typename T> SYCL_EXTERNAL T div(T a, T b) { return a / b; }
template <typename T> SYCL_EXTERNAL T neg(T a) { return -a; }
template <typename T> SYCL_EXTERNAL T pre_inc(T a) { return ++a; }
template <typename T> SYCL_EXTERNAL T pre_dec(T a) { return --a; }
template <typename T> SYCL_EXTERNAL T fma(T a, T b, T c) { return a * b + c; }
template <typename T> SYCL_EXTERNAL T exp(T a) { return sycl::exp(a); }

#define TEST_TYPE(type)                               \
  template type add(type a, type b);                  \
  template type sub(type a, type b);                  \
  template type mul(type a, type b);                  \
  template type div(type a, type b);                  \
  template type neg(type a);                          \
  template type pre_inc(type a);                      \
  template type pre_dec(type a);                      \
  template type fma(type a, type b, type c);          \
  template type exp(type a);                          \

// COM: No point in testing more than one fp type for now.

TEST_TYPE(float)

// CHECK-LABEL:     func.func @_Z3addIfET_S0_S0_(
// CHECK-SAME:                                   %[[VAL_151:.*]]: f32 {llvm.noundef}, %[[VAL_152:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:   %[[VAL_153:.*]] = arith.addf %[[VAL_151]], %[[VAL_152]] : f32
// CHECK-FAST:        %[[VAL_153:.*]] = arith.addf %[[VAL_151]], %[[VAL_152]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_153:.*]] = arith.addf %[[VAL_151]], %[[VAL_152]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_153]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z3subIfET_S0_S0_(
// CHECK-SAME:                                   %[[VAL_154:.*]]: f32 {llvm.noundef}, %[[VAL_155:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:        %[[VAL_156:.*]] = arith.subf %[[VAL_154]], %[[VAL_155]] : f32
// CHECK-FAST:             %[[VAL_156:.*]] = arith.subf %[[VAL_154]], %[[VAL_155]] fastmath<fast> : f32
// CHECK-NOHONOR:          %[[VAL_156:.*]] = arith.subf %[[VAL_154]], %[[VAL_155]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_156]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z3mulIfET_S0_S0_(
// CHECK-SAME:                                   %[[VAL_157:.*]]: f32 {llvm.noundef}, %[[VAL_158:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:   %[[VAL_159:.*]] = arith.mulf %[[VAL_157]], %[[VAL_158]] : f32
// CHECK-FAST:        %[[VAL_159:.*]] = arith.mulf %[[VAL_157]], %[[VAL_158]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_159:.*]] = arith.mulf %[[VAL_157]], %[[VAL_158]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_159]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z3divIfET_S0_S0_(
// CHECK-SAME:                                   %[[VAL_160:.*]]: f32 {llvm.noundef}, %[[VAL_161:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:   %[[VAL_162:.*]] = arith.divf %[[VAL_160]], %[[VAL_161]] : f32
// CHECK-FAST:        %[[VAL_162:.*]] = arith.divf %[[VAL_160]], %[[VAL_161]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_162:.*]] = arith.divf %[[VAL_160]], %[[VAL_161]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_162]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z3negIfET_S0_(
// CHECK-SAME:                                %[[VAL_163:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:   %[[VAL_164:.*]] = arith.negf %[[VAL_163]] : f32
// CHECK-FAST:        %[[VAL_164:.*]] = arith.negf %[[VAL_163]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_164:.*]] = arith.negf %[[VAL_163]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_164]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z7pre_incIfET_S0_(
// CHECK-SAME:                                    %[[VAL_165:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK:             %[[VAL_166:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-PRESTRICT:   %[[VAL_167:.*]] = arith.addf %[[VAL_165]], %[[VAL_166]] : f32
// CHECK-FAST:        %[[VAL_167:.*]] = arith.addf %[[VAL_165]], %[[VAL_166]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_167:.*]] = arith.addf %[[VAL_165]], %[[VAL_166]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_167]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z7pre_decIfET_S0_(
// CHECK-SAME:                                    %[[VAL_169:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK:             %[[VAL_170:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-PRESTRICT:   %[[VAL_171:.*]] = arith.subf %[[VAL_169]], %[[VAL_170]] : f32
// CHECK-FAST:        %[[VAL_171:.*]] = arith.subf %[[VAL_169]], %[[VAL_170]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_171:.*]] = arith.subf %[[VAL_169]], %[[VAL_170]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_171]] : f32
// CHECK:           }

// CHECK:           func.func @_Z3fmaIfET_S0_S0_S0_(
// CHECK-SAME:                                      %[[VAL_175:.*]]: f32 {llvm.noundef}, %[[VAL_176:.*]]: f32 {llvm.noundef}, %[[VAL_177:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRECISE:     %[[VAL_178:.*]] = math.fma %[[VAL_175]], %[[VAL_176]], %[[VAL_177]] : f32
// CHECK-FAST:        %[[ADD:.*]] = arith.mulf %[[VAL_175]], %[[VAL_176]] fastmath<fast> : f32
// CHECK-FAST:        %[[VAL_178:.*]] = arith.addf %[[ADD]], %[[VAL_177]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_178:.*]] = math.fma %[[VAL_175]], %[[VAL_176]], %[[VAL_177]] fastmath<nnan,ninf> : f32
// CHECK-STRICT:      %[[ADD:.*]] = arith.mulf %[[VAL_175]], %[[VAL_176]] : f32
// CHECK-STRICT:      %[[VAL_178:.*]] = arith.addf %[[ADD]], %[[VAL_177]] : f32
// CHECK:             return %[[VAL_178]] : f32
// CHECK:           }

// CHECK-LABEL:     func.func @_Z3expIfET_S0_(
// CHECK-SAME:                                %[[VAL_179:.*]]: f32 {llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-PRESTRICT:   %[[VAL_180:.*]] = sycl.math.exp %[[VAL_179]] : f32
// CHECK-FAST:        %[[VAL_180:.*]] = sycl.math.exp %[[VAL_179]] fastmath<fast> : f32
// CHECK-NOHONOR:     %[[VAL_180:.*]] = sycl.math.exp %[[VAL_179]] fastmath<nnan,ninf> : f32
// CHECK:             return %[[VAL_180]] : f32
// CHECK:           }
