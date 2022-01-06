//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

  #include <clcmacro.h>
  #include <spirv/spirv.h>
 
  void __ocml_sincos_f64(double, double *, double *);
  void __ocml_sincos_f32(float, float *, float * ); 

  #define __CLC_SINCOS_IMPL(ADDRSPACE, BUILTIN, FP_TYPE, ARG_TYPE)               \
    _CLC_OVERLOAD _CLC_DEF ARG_TYPE __spirv_ocl_sincos(                          \
        ARG_TYPE x, ADDRSPACE ARG_TYPE *cosval_ptr) {                            \
      FP_TYPE sinval;                                                            \
      FP_TYPE cosval;                                                            \
      BUILTIN(x, &sinval, &cosval);                                              \
      *cosval_ptr = cosval;                                                      \
      return sinval;                                                             \
    }
  
  #define __CLC_SINCOS(BUILTIN, FP_TYPE, ARG_TYPE)                               \
    __CLC_SINCOS_IMPL(global, BUILTIN, FP_TYPE, ARG_TYPE)                        \
    __CLC_SINCOS_IMPL(local, BUILTIN, FP_TYPE, ARG_TYPE)                         \
    __CLC_SINCOS_IMPL(private, BUILTIN, FP_TYPE, ARG_TYPE)
  
  __CLC_SINCOS(__ocml_sincos_f32, float, float)
  
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_sincos, float,
                        private, float)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_sincos, float,
                        local, float)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_sincos, float,
                        global, float)
  
  #ifdef cl_khr_fp64
  __CLC_SINCOS(__ocml_sincos_f64, double, double)
  
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_sincos,
                        double, private, double)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_sincos,
                        double, local, double)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_sincos,
                        double, global, double)
  #endif
  
  #ifdef cl_khr_fp16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __CLC_SINCOS(__ocml_sincos_f32, float, half)
  
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_sincos, half,
                        private, half)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_sincos, half,
                        local, half)
  _CLC_V_V_VP_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_sincos, half,
                        global, half)
  #endif
