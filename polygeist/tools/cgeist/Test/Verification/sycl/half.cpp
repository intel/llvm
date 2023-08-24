// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK: !sycl_half = !sycl.half<(f16)>

// CHECK-LABEL:     func.func @_Z8identityN4sycl3_V16detail9half_impl4halfE(
// CHECK-SAME:                                                              %[[VAL_151:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_152:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_153:.*]] = memref.cast %[[VAL_152]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_154:.*]] = memref.memory_space_cast %[[VAL_153]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_155:.*]] = memref.memory_space_cast %[[VAL_151]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_154]], %[[VAL_155]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1EOS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_156:.*]] = affine.load %[[VAL_152]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        return %[[VAL_156]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half identity(sycl::half h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z8identityN4sycl3_V13vecINS0_6detail9half_impl4halfELi8EEE(
// CHECK-SAME:                                                                             %[[VAL_163:.*]]: memref<?x!sycl_vec_sycl_half_8_> {llvm.align = 16 : i64, llvm.byval = !sycl_vec_sycl_half_8_, llvm.noundef}) -> !sycl_vec_sycl_half_8_
// CHECK-NEXT:        %[[VAL_164:.*]] = memref.alloca() : memref<1x!sycl_vec_sycl_half_8_>
// CHECK-NEXT:        %[[VAL_165:.*]] = memref.cast %[[VAL_164]] : memref<1x!sycl_vec_sycl_half_8_> to memref<?x!sycl_vec_sycl_half_8_>
// CHECK-NEXT:        %[[VAL_166:.*]] = memref.memory_space_cast %[[VAL_165]] : memref<?x!sycl_vec_sycl_half_8_> to memref<?x!sycl_vec_sycl_half_8_, 4>
// CHECK-NEXT:        %[[VAL_167:.*]] = memref.memory_space_cast %[[VAL_163]] : memref<?x!sycl_vec_sycl_half_8_> to memref<?x!sycl_vec_sycl_half_8_, 4>
// CHECK-NEXT:        sycl.constructor @vec(%[[VAL_166]], %[[VAL_167]]) {MangledFunctionName = @_ZN4sycl3_V13vecINS0_6detail9half_impl4halfELi8EEC1EOS5_} : (memref<?x!sycl_vec_sycl_half_8_, 4>, memref<?x!sycl_vec_sycl_half_8_, 4>)
// CHECK-NEXT:        %[[VAL_168:.*]] = affine.load %[[VAL_164]][0] : memref<1x!sycl_vec_sycl_half_8_>
// CHECK-NEXT:        return %[[VAL_168]] : !sycl_vec_sycl_half_8_
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half8 identity(sycl::half8 h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z6toHalff(
// CHECK-SAME:                            %[[VAL_163:.*]]: f32 {llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_164:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_165:.*]] = memref.cast %[[VAL_164]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_166:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:        %[[VAL_167:.*]] = memref.cast %[[VAL_166]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:        %[[VAL_168:.*]] = llvm.mlir.undef : f32
// CHECK-NEXT:        affine.store %[[VAL_163]], %[[VAL_166]][0] : memref<1xf32>
// CHECK-NEXT:        %[[VAL_169:.*]] = memref.memory_space_cast %[[VAL_165]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_170:.*]] = memref.memory_space_cast %[[VAL_167]] : memref<?xf32> to memref<?xf32, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_169]], %[[VAL_170]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKf} : (memref<?x!sycl_half, 4>, memref<?xf32, 4>)
// CHECK-NEXT:        %[[VAL_171:.*]] = affine.load %[[VAL_164]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        return %[[VAL_171]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half toHalf(float h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z6toHalfd(
// CHECK-SAME:                            %[[VAL_178:.*]]: f64 {llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_179:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:        %[[VAL_180:.*]] = memref.cast %[[VAL_179]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:        %[[VAL_181:.*]] = llvm.mlir.undef : f32
// CHECK-NEXT:        affine.store %[[VAL_181]], %[[VAL_179]][0] : memref<1xf32>
// CHECK-NEXT:        %[[VAL_182:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_183:.*]] = memref.cast %[[VAL_182]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_184:.*]] = arith.truncf %[[VAL_178]] : f64 to f32
// CHECK-NEXT:        %[[VAL_185:.*]] = memref.memory_space_cast %[[VAL_180]] : memref<?xf32> to memref<?xf32, 4>
// CHECK-NEXT:        affine.store %[[VAL_184]], %[[VAL_185]][0] : memref<?xf32, 4>
// CHECK-NEXT:        %[[VAL_186:.*]] = memref.memory_space_cast %[[VAL_183]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_186]], %[[VAL_185]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKf} : (memref<?x!sycl_half, 4>, memref<?xf32, 4>)
// CHECK-NEXT:        %[[VAL_187:.*]] = affine.load %[[VAL_182]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        return %[[VAL_187]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half toHalf(double h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z6toHalfi(
// CHECK-SAME:                            %[[VAL_188:.*]]: i32 {llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_189:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:        %[[VAL_190:.*]] = memref.cast %[[VAL_189]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:        %[[VAL_191:.*]] = llvm.mlir.undef : f32
// CHECK-NEXT:        affine.store %[[VAL_191]], %[[VAL_189]][0] : memref<1xf32>
// CHECK-NEXT:        %[[VAL_192:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_193:.*]] = memref.cast %[[VAL_192]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_194:.*]] = arith.sitofp %[[VAL_188]] : i32 to f32
// CHECK-NEXT:        %[[VAL_195:.*]] = memref.memory_space_cast %[[VAL_190]] : memref<?xf32> to memref<?xf32, 4>
// CHECK-NEXT:        affine.store %[[VAL_194]], %[[VAL_195]][0] : memref<?xf32, 4>
// CHECK-NEXT:        %[[VAL_196:.*]] = memref.memory_space_cast %[[VAL_193]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_196]], %[[VAL_195]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKf} : (memref<?x!sycl_half, 4>, memref<?xf32, 4>)
// CHECK-NEXT:        %[[VAL_197:.*]] = affine.load %[[VAL_192]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        return %[[VAL_197]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half toHalf(int h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z6toHalfb(
// CHECK-SAME:                            %[[VAL_197:.*]]: i1 {llvm.noundef, llvm.zeroext}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_198:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT:        %[[VAL_199:.*]] = memref.cast %[[VAL_198]] : memref<1xf32> to memref<?xf32>
// CHECK-NEXT:        %[[VAL_200:.*]] = llvm.mlir.undef : f32
// CHECK-NEXT:        affine.store %[[VAL_200]], %[[VAL_198]][0] : memref<1xf32>
// CHECK-NEXT:        %[[VAL_201:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_202:.*]] = memref.cast %[[VAL_201]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_203:.*]] = arith.uitofp %[[VAL_197]] : i1 to f32
// CHECK-NEXT:        %[[VAL_204:.*]] = memref.memory_space_cast %[[VAL_199]] : memref<?xf32> to memref<?xf32, 4>
// CHECK-NEXT:        affine.store %[[VAL_203]], %[[VAL_204]][0] : memref<?xf32, 4>
// CHECK-NEXT:        %[[VAL_205:.*]] = memref.memory_space_cast %[[VAL_202]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_205]], %[[VAL_204]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKf} : (memref<?x!sycl_half, 4>, memref<?xf32, 4>)
// CHECK-NEXT:        %[[VAL_206:.*]] = affine.load %[[VAL_201]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        return %[[VAL_206]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half toHalf(bool h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z7toFloatN4sycl3_V16detail9half_impl4halfE(
// CHECK-SAME:                                                             %[[VAL_208:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> (f32 {llvm.noundef})
// CHECK-NEXT:        %[[VAL_209:.*]] = memref.memory_space_cast %[[VAL_208]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_210:.*]] = sycl.call @"operator float"(%[[VAL_209]]) {MangledFunctionName = @_ZNK4sycl3_V16detail9half_impl4halfcvfEv, TypeName = @half} : (memref<?x!sycl_half, 4>) -> f32
// CHECK-NEXT:        return %[[VAL_210]] : f32
// CHECK-NEXT:      }

SYCL_EXTERNAL float toFloat(sycl::half h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z8toDoubleN4sycl3_V16detail9half_impl4halfE(
// CHECK-SAME:                                                              %[[VAL_216:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> (f64 {llvm.noundef})
// CHECK-NEXT:        %[[VAL_217:.*]] = memref.memory_space_cast %[[VAL_216]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_218:.*]] = sycl.call @"operator float"(%[[VAL_217]]) {MangledFunctionName = @_ZNK4sycl3_V16detail9half_impl4halfcvfEv, TypeName = @half} : (memref<?x!sycl_half, 4>) -> f32
// CHECK-NEXT:        %[[VAL_219:.*]] = arith.extf %[[VAL_218]] : f32 to f64
// CHECK-NEXT:        return %[[VAL_219]] : f64
// CHECK-NEXT:      }

SYCL_EXTERNAL double toDouble(sycl::half h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z5toIntN4sycl3_V16detail9half_impl4halfE(
// CHECK-SAME:                                                           %[[VAL_220:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> (i32 {llvm.noundef})
// CHECK-NEXT:        %[[VAL_221:.*]] = memref.memory_space_cast %[[VAL_220]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_222:.*]] = sycl.call @"operator float"(%[[VAL_221]]) {MangledFunctionName = @_ZNK4sycl3_V16detail9half_impl4halfcvfEv, TypeName = @half} : (memref<?x!sycl_half, 4>) -> f32
// CHECK-NEXT:        %[[VAL_223:.*]] = arith.fptosi %[[VAL_222]] : f32 to i32
// CHECK-NEXT:        return %[[VAL_223]] : i32
// CHECK-NEXT:      }

SYCL_EXTERNAL int toInt(sycl::half h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z6toBoolN4sycl3_V16detail9half_impl4halfE(
// CHECK-SAME:                                                            %[[VAL_224:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext})
// CHECK-NEXT:        %[[VAL_225:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %[[VAL_226:.*]] = memref.memory_space_cast %[[VAL_224]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_227:.*]] = sycl.call @"operator float"(%[[VAL_226]]) {MangledFunctionName = @_ZNK4sycl3_V16detail9half_impl4halfcvfEv, TypeName = @half} : (memref<?x!sycl_half, 4>) -> f32
// CHECK-NEXT:        %[[VAL_228:.*]] = arith.cmpf une, %[[VAL_227]], %[[VAL_225]] : f32
// CHECK-NEXT:        return %[[VAL_228]] : i1
// CHECK-NEXT:      }

SYCL_EXTERNAL bool toBool(sycl::half h) {
  return h;
}

// CHECK-LABEL:     func.func @_Z3addN4sycl3_V16detail9half_impl4halfES3_(
// CHECK-SAME:                                                            %[[VAL_240:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}, %[[VAL_241:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_242:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_243:.*]] = memref.cast %[[VAL_242]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_244:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_245:.*]] = memref.cast %[[VAL_244]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_246:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_247:.*]] = memref.cast %[[VAL_246]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_248:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_249:.*]] = memref.cast %[[VAL_248]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_250:.*]] = memref.memory_space_cast %[[VAL_249]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_251:.*]] = memref.memory_space_cast %[[VAL_240]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_250]], %[[VAL_251]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_252:.*]] = memref.memory_space_cast %[[VAL_247]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_253:.*]] = memref.memory_space_cast %[[VAL_241]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_252]], %[[VAL_253]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_254:.*]] = affine.load %[[VAL_248]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_254]], %[[VAL_244]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_255:.*]] = affine.load %[[VAL_246]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_255]], %[[VAL_242]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_256:.*]] = sycl.call @"operator+"(%[[VAL_245]], %[[VAL_243]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_implplENS2_4halfES3_} : (memref<?x!sycl_half>, memref<?x!sycl_half>) -> !sycl_half
// CHECK-NEXT:        return %[[VAL_256]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half add(sycl::half lhs, sycl::half rhs) {
  return lhs + rhs;
}

// CHECK-LABEL:           func.func @_Z3addN4sycl3_V13vecINS0_6detail9half_impl4halfELi4EEES5_(
// CHECK-SAME:                                                                                 %[[VAL_278:.*]]: memref<?x!sycl_vec_sycl_half_4_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_4_, llvm.noundef}, %[[VAL_279:.*]]: memref<?x!sycl_vec_sycl_half_4_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_4_, llvm.noundef}) -> !sycl_vec_sycl_half_4_
// CHECK-NEXT:        %[[VAL_280:.*]] = memref.memory_space_cast %[[VAL_278]] : memref<?x!sycl_vec_sycl_half_4_> to memref<?x!sycl_vec_sycl_half_4_, 4>
// CHECK-NEXT:        %[[VAL_281:.*]] = memref.memory_space_cast %[[VAL_279]] : memref<?x!sycl_vec_sycl_half_4_> to memref<?x!sycl_vec_sycl_half_4_, 4>
// CHECK-NEXT:        %[[VAL_282:.*]] = sycl.call @"operator+"(%[[VAL_280]], %[[VAL_281]]) {MangledFunctionName = @_ZNK4sycl3_V13vecINS0_6detail9half_impl4halfELi4EEplIS5_EES5_RKNSt9enable_ifILb1ET_E4typeE, TypeName = @vec} : (memref<?x!sycl_vec_sycl_half_4_, 4>, memref<?x!sycl_vec_sycl_half_4_, 4>) -> !sycl_vec_sycl_half_4_
// CHECK-NEXT:        return %[[VAL_282]] : !sycl_vec_sycl_half_4_
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half4 add(sycl::half4 lhs, sycl::half4 rhs) {
  return lhs + rhs;
}

// CHECK-LABEL:     func.func @_Z3subN4sycl3_V16detail9half_impl4halfES3_(
// CHECK-SAME:                                                            %[[VAL_275:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}, %[[VAL_276:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_277:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_278:.*]] = memref.cast %[[VAL_277]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_279:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_280:.*]] = memref.cast %[[VAL_279]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_281:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_282:.*]] = memref.cast %[[VAL_281]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_283:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_284:.*]] = memref.cast %[[VAL_283]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_285:.*]] = memref.memory_space_cast %[[VAL_284]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_286:.*]] = memref.memory_space_cast %[[VAL_275]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_285]], %[[VAL_286]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_287:.*]] = memref.memory_space_cast %[[VAL_282]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_288:.*]] = memref.memory_space_cast %[[VAL_276]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_287]], %[[VAL_288]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_289:.*]] = affine.load %[[VAL_283]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_289]], %[[VAL_279]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_290:.*]] = affine.load %[[VAL_281]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_290]], %[[VAL_277]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_291:.*]] = sycl.call @"operator-"(%[[VAL_280]], %[[VAL_278]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_implmiENS2_4halfES3_} : (memref<?x!sycl_half>, memref<?x!sycl_half>) -> !sycl_half
// CHECK-NEXT:        return %[[VAL_291]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half sub(sycl::half lhs, sycl::half rhs) {
  return lhs - rhs;
}

// CHECK-LABEL:     func.func @_Z3subN4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEES5_(
// CHECK-SAME:                                                                           %[[VAL_330:.*]]: memref<?x!sycl_vec_sycl_half_2_> {llvm.align = 4 : i64, llvm.byval = !sycl_vec_sycl_half_2_, llvm.noundef}, %[[VAL_331:.*]]: memref<?x!sycl_vec_sycl_half_2_> {llvm.align = 4 : i64, llvm.byval = !sycl_vec_sycl_half_2_, llvm.noundef}) -> !sycl_vec_sycl_half_2_
// CHECK-NEXT:        %[[VAL_332:.*]] = memref.memory_space_cast %[[VAL_330]] : memref<?x!sycl_vec_sycl_half_2_> to memref<?x!sycl_vec_sycl_half_2_, 4>
// CHECK-NEXT:        %[[VAL_333:.*]] = memref.memory_space_cast %[[VAL_331]] : memref<?x!sycl_vec_sycl_half_2_> to memref<?x!sycl_vec_sycl_half_2_, 4>
// CHECK-NEXT:        %[[VAL_334:.*]] = sycl.call @"operator-"(%[[VAL_332]], %[[VAL_333]]) {MangledFunctionName = @_ZNK4sycl3_V13vecINS0_6detail9half_impl4halfELi2EEmiIS5_EES5_RKNSt9enable_ifILb1ET_E4typeE, TypeName = @vec} : (memref<?x!sycl_vec_sycl_half_2_, 4>, memref<?x!sycl_vec_sycl_half_2_, 4>) -> !sycl_vec_sycl_half_2_
// CHECK-NEXT:        return %[[VAL_334]] : !sycl_vec_sycl_half_2_
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half2 sub(sycl::half2 lhs, sycl::half2 rhs) {
  return lhs - rhs;
}

// CHECK-LABEL:     func.func @_Z3mulN4sycl3_V16detail9half_impl4halfES3_(
// CHECK-SAME:                                                            %[[VAL_304:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}, %[[VAL_305:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_306:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_307:.*]] = memref.cast %[[VAL_306]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_308:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_309:.*]] = memref.cast %[[VAL_308]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_310:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_311:.*]] = memref.cast %[[VAL_310]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_312:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_313:.*]] = memref.cast %[[VAL_312]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_314:.*]] = memref.memory_space_cast %[[VAL_313]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_315:.*]] = memref.memory_space_cast %[[VAL_304]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_314]], %[[VAL_315]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_316:.*]] = memref.memory_space_cast %[[VAL_311]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_317:.*]] = memref.memory_space_cast %[[VAL_305]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_316]], %[[VAL_317]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_318:.*]] = affine.load %[[VAL_312]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_318]], %[[VAL_308]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_319:.*]] = affine.load %[[VAL_310]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_319]], %[[VAL_306]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_320:.*]] = sycl.call @"operator*"(%[[VAL_309]], %[[VAL_307]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_implmlENS2_4halfES3_} : (memref<?x!sycl_half>, memref<?x!sycl_half>) -> !sycl_half
// CHECK-NEXT:        return %[[VAL_320]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half mul(sycl::half lhs, sycl::half rhs) {
  return lhs * rhs;
}

// CHECK-LABEL:     func.func @_Z3mulN4sycl3_V13vecINS0_6detail9half_impl4halfELi4EEES5_(
// CHECK-SAME:                                                                           %[[VAL_382:.*]]: memref<?x!sycl_vec_sycl_half_4_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_4_, llvm.noundef}, %[[VAL_383:.*]]: memref<?x!sycl_vec_sycl_half_4_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_4_, llvm.noundef}) -> !sycl_vec_sycl_half_4_
// CHECK-NEXT:        %[[VAL_384:.*]] = memref.memory_space_cast %[[VAL_382]] : memref<?x!sycl_vec_sycl_half_4_> to memref<?x!sycl_vec_sycl_half_4_, 4>
// CHECK-NEXT:        %[[VAL_385:.*]] = memref.memory_space_cast %[[VAL_383]] : memref<?x!sycl_vec_sycl_half_4_> to memref<?x!sycl_vec_sycl_half_4_, 4>
// CHECK-NEXT:        %[[VAL_386:.*]] = sycl.call @"operator*"(%[[VAL_384]], %[[VAL_385]]) {MangledFunctionName = @_ZNK4sycl3_V13vecINS0_6detail9half_impl4halfELi4EEmlIS5_EES5_RKNSt9enable_ifILb1ET_E4typeE, TypeName = @vec} : (memref<?x!sycl_vec_sycl_half_4_, 4>, memref<?x!sycl_vec_sycl_half_4_, 4>) -> !sycl_vec_sycl_half_4_
// CHECK-NEXT:        return %[[VAL_386]] : !sycl_vec_sycl_half_4_
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half4 mul(sycl::half4 lhs, sycl::half4 rhs) {
  return lhs * rhs;
}

// CHECK-LABEL:     func.func @_Z3divN4sycl3_V16detail9half_impl4halfES3_(
// CHECK-SAME:                                                            %[[VAL_321:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}, %[[VAL_322:.*]]: memref<?x!sycl_half> {llvm.align = 2 : i64, llvm.byval = !sycl_half, llvm.noundef}) -> !sycl_half
// CHECK-NEXT:        %[[VAL_323:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_324:.*]] = memref.cast %[[VAL_323]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_325:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_326:.*]] = memref.cast %[[VAL_325]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_327:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_328:.*]] = memref.cast %[[VAL_327]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_329:.*]] = memref.alloca() : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_330:.*]] = memref.cast %[[VAL_329]] : memref<1x!sycl_half> to memref<?x!sycl_half>
// CHECK-NEXT:        %[[VAL_331:.*]] = memref.memory_space_cast %[[VAL_330]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_332:.*]] = memref.memory_space_cast %[[VAL_321]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_331]], %[[VAL_332]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_333:.*]] = memref.memory_space_cast %[[VAL_328]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        %[[VAL_334:.*]] = memref.memory_space_cast %[[VAL_322]] : memref<?x!sycl_half> to memref<?x!sycl_half, 4>
// CHECK-NEXT:        sycl.constructor @half(%[[VAL_333]], %[[VAL_334]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impl4halfC1ERKS3_} : (memref<?x!sycl_half, 4>, memref<?x!sycl_half, 4>)
// CHECK-NEXT:        %[[VAL_335:.*]] = affine.load %[[VAL_329]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_335]], %[[VAL_325]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_336:.*]] = affine.load %[[VAL_327]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        affine.store %[[VAL_336]], %[[VAL_323]][0] : memref<1x!sycl_half>
// CHECK-NEXT:        %[[VAL_337:.*]] = sycl.call @"operator/"(%[[VAL_326]], %[[VAL_324]]) {MangledFunctionName = @_ZN4sycl3_V16detail9half_impldvENS2_4halfES3_} : (memref<?x!sycl_half>, memref<?x!sycl_half>) -> !sycl_half
// CHECK-NEXT:        return %[[VAL_337]] : !sycl_half
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half div(sycl::half lhs, sycl::half rhs) {
  return lhs / rhs;
}

// CHECK-LABEL:     func.func @_Z3divN4sycl3_V13vecINS0_6detail9half_impl4halfELi3EEES5_(
// CHECK-SAME:                                                                           %[[VAL_434:.*]]: memref<?x!sycl_vec_sycl_half_3_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_3_, llvm.noundef}, %[[VAL_435:.*]]: memref<?x!sycl_vec_sycl_half_3_> {llvm.align = 8 : i64, llvm.byval = !sycl_vec_sycl_half_3_, llvm.noundef}) -> !sycl_vec_sycl_half_3_
// CHECK-NEXT:        %[[VAL_436:.*]] = memref.memory_space_cast %[[VAL_434]] : memref<?x!sycl_vec_sycl_half_3_> to memref<?x!sycl_vec_sycl_half_3_, 4>
// CHECK-NEXT:        %[[VAL_437:.*]] = memref.memory_space_cast %[[VAL_435]] : memref<?x!sycl_vec_sycl_half_3_> to memref<?x!sycl_vec_sycl_half_3_, 4>
// CHECK-NEXT:        %[[VAL_438:.*]] = sycl.call @"operator/"(%[[VAL_436]], %[[VAL_437]]) {MangledFunctionName = @_ZNK4sycl3_V13vecINS0_6detail9half_impl4halfELi3EEdvIS5_EES5_RKNSt9enable_ifILb1ET_E4typeE, TypeName = @vec} : (memref<?x!sycl_vec_sycl_half_3_, 4>, memref<?x!sycl_vec_sycl_half_3_, 4>) -> !sycl_vec_sycl_half_3_
// CHECK-NEXT:        return %[[VAL_438]] : !sycl_vec_sycl_half_3_
// CHECK-NEXT:      }

SYCL_EXTERNAL sycl::half3 div(sycl::half3 lhs, sycl::half3 rhs) {
  return lhs / rhs;
}
