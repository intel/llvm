===================================================================================
SPIR-V representation in LLVM IR for FP8, FP4 and Int4 datatypes
===================================================================================
.. contents::
   :local:

Overview
========

Open Compute and other projects are adding various new data-types and SPIR-V
(starting from SPV_EXT_float8) is now adopting them. None of these data-types
have appropriate LLVM IR counterparts. This document describes the proposed
LLVM IR input format for *FP8*, *FP4*, and *Int4* types, the translation flow, and the
expected LLVM IR output from the consumer.

SPIR-V Non-standard Types Mapped to LLVM Types
==============================================

All formats of *FP8* (E4M3, E5M2), *FP4* (E2M1), and *Int4* will be represented in LLVM IR with
integer types (*i8* for FP8, *i4* for FP4 and Int4).
Until 'type resolution' instruction appears in the module (see below), these values will
remain as integers. When 'type resolution' instruction is being processed, integer values will be bitcasted
to floating-point or integer values with appropriate width and encoding depending on the instruction. If the instruction's
result is *FP8*, *FP4*, *Int4*, or a composite containing them, then it is also being bitcasted to the
appropriate integer type or composite. It is safe to do as these extensions don't add support
for arithmetic instructions and builtins (unless it's *OpCooperativeMatrixMulAddKHR*, but this
case will be handled separately).

The 'type resolution' instruction can be either a conversion instruction or *OpCooperativeMatrixMulAddKHR*.

**Type mappings:**

* FP8 (E4M3, E5M2) → *i8*
* FP4 (E2M1) → *i4*
* Int4 → *i4*

SPIR-V conversion instructions
==============================

Most conversions will be represented by standard SPIR-V conversion instructions (*OpFConvert*, *OpConvertSToF*, *OpConvertFToS*,
*OpConvertUToF*, *OpConvertFToU*, *OpSConvert*), which don't carry information about floating-point value's width and encoding.
This document adds a new set of external function calls, each of which has a name that is formed from encoding a specific conversion
that it performs. This name has a *__builtin_spirv_* prefix and a postfix indicating the extension (e.g., *EXT* from SPV_EXT_float8,
*INTEL* from SPV_INTEL_int4/SPV_INTEL_float4/SPV_INTEL_fp_conversions). These calls will be translated to SPIR-V conversion
instructions operating over the appropriate types. These functions are expected to be mangled following Itanium C++ ABI. SPIR-V consumer
will apply Itanium mangling during translation to LLVM IR as well.

SPIR-V generator will support *scalar*, *vector* and *packed* for the conversion builtin functions as LLVM IR input;
*packed* format is translated to a *vector*. Meanwhile SPIR-V consumer will never pack a *vector* back to *packed* format.

SPV_EXT_float8 and SPV_KHR_bfloat16 Conversions
------------------------------------------------

**Translated to OpFConvert:**

.. code-block:: C

  __builtin_spirv_ConvertFP16ToE4M3EXT, __builtin_spirv_ConvertBF16ToE4M3EXT,
  __builtin_spirv_ConvertFP16ToE5M2EXT, __builtin_spirv_ConvertBF16ToE5M2EXT,
  __builtin_spirv_ConvertE4M3ToFP16EXT, __builtin_spirv_ConvertE5M2ToFP16EXT,
  __builtin_spirv_ConvertE4M3ToBF16EXT, __builtin_spirv_ConvertE5M2ToBF16EXT

SPV_INTEL_int4 Conversions
---------------------------

**Translated to OpConvertSToF:**

.. code-block:: C

  __builtin_spirv_ConvertInt4ToE4M3INTEL, __builtin_spirv_ConvertInt4ToE5M2INTEL,
  __builtin_spirv_ConvertInt4ToFP16INTEL, __builtin_spirv_ConvertInt4ToBF16INTEL

**Translated to OpConvertFToS:**

.. code-block:: C

  __builtin_spirv_ConvertFP16ToInt4INTEL, __builtin_spirv_ConvertBF16ToInt4INTEL

**Translated to OpSConvert:**

.. code-block:: C

  __builtin_spirv_ConvertInt4ToInt8INTEL

SPV_INTEL_float4 Conversions
-----------------------------

**Translated to OpFConvert:**

.. code-block:: C

  __builtin_spirv_ConvertE2M1ToE4M3INTEL, __builtin_spirv_ConvertE2M1ToE5M2INTEL,
  __builtin_spirv_ConvertE2M1ToFP16INTEL, __builtin_spirv_ConvertE2M1ToBF16INTEL,
  __builtin_spirv_ConvertFP16ToE2M1INTEL, __builtin_spirv_ConvertBF16ToE2M1INTEL

SPV_INTEL_fp_conversions
-------------------------

This extension provides conversions with specialized rounding modes for improved precision and efficiency.

**Translated to OpClampConvertFToFINTEL (clamp rounding):**

.. code-block:: C

  __builtin_spirv_ClampConvertFP16ToE2M1INTEL, __builtin_spirv_ClampConvertBF16ToE2M1INTEL,
  __builtin_spirv_ClampConvertFP16ToE4M3INTEL, __builtin_spirv_ClampConvertBF16ToE4M3INTEL,
  __builtin_spirv_ClampConvertFP16ToE5M2INTEL, __builtin_spirv_ClampConvertBF16ToE5M2INTEL

**Translated to OpClampConvertFToSINTEL (clamp rounding to signed integer):**

.. code-block:: C

  __builtin_spirv_ClampConvertFP16ToInt4INTEL, __builtin_spirv_ClampConvertBF16ToInt4INTEL

**Translated to OpStochasticRoundFToFINTEL (stochastic rounding):**

.. code-block:: C

  __builtin_spirv_StochasticRoundFP16ToE5M2INTEL, __builtin_spirv_StochasticRoundFP16ToE4M3INTEL,
  __builtin_spirv_StochasticRoundBF16ToE5M2INTEL, __builtin_spirv_StochasticRoundBF16ToE4M3INTEL,
  __builtin_spirv_StochasticRoundFP16ToE2M1INTEL, __builtin_spirv_StochasticRoundBF16ToE2M1INTEL

Note: These functions take an additional seed parameter (i32) and may optionally take a pointer parameter
for storing the last seed value.

**Translated to OpClampStochasticRoundFToSINTEL (clamp + stochastic rounding to signed integer):**

.. code-block:: C

  __builtin_spirv_ClampStochasticRoundFP16ToInt4INTEL, __builtin_spirv_ClampStochasticRoundBF16ToInt4INTEL

**Translated to OpClampStochasticRoundFToFINTEL (clamp + stochastic rounding):**

.. code-block:: C

  __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL, __builtin_spirv_ClampStochasticRoundFP16ToE4M3INTEL,
  __builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL, __builtin_spirv_ClampStochasticRoundBF16ToE4M3INTEL


Example LLVM IR to SPIR-V translation:
Input LLVM IR

.. code-block:: C

   %alloc = alloca half
   %FP16_val = call half __builtin_spirv_ConvertE4M3ToFP16EXT(i8 1)
   store half %FP16_val, ptr %alloc

Output SPIR-V

.. code-block:: C

   %half_ty = OpTypeFloat 16 0
   %ptr_ty = OpTypePointer %half_ty Private
   %int8_ty = OpTypeInt 8 0
   %fp8_ty = OpTypeFloat 8 1
   %const = OpConstant %int8_ty 1
   /*...*/
   %alloc = OpVariable %half_ty Private
   %fp8_val = OpBitCast %fp8_ty %const
   %fp16_val = OpFConvert %half_ty %fp8_val
   OpStore %fp16_val %alloc

Output LLVM IR

.. code-block:: C

   %alloc = alloca half
   %fp16_val = call half __builtin_spirv_ConvertE4M3ToFP16EXT(i8 1)
   store half %fp16_val, ptr %alloc

SPIR-V cooperative matrix instructions
======================================

TBD
