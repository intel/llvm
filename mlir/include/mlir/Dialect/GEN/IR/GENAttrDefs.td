//===-- GENAttrDefs.td - GEN dialect attributes def. file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEN_ATTRDEFS
#define GEN_ATTRDEFS

include "mlir/IR/EnumAttr.td"

/// Enum attribute of the different shuffle kinds.
def GEN_ShflKindAttr : I32EnumAttr<"ShflKind", "GEN shuffle kind",
  [
    I32EnumAttrCase<"XOR",  0, "xor">,
    I32EnumAttrCase<"UP",   1, "up">,
    I32EnumAttrCase<"DOWN", 2, "down">,
    I32EnumAttrCase<"IDX",  3, "idx">
  ]> {
  let cppNamespace = "::mlir::GEN";
}

/// Enum attribute of the different floating-point rounding modes.
def GEN_RoundingModeAttr : I32EnumAttr<"RoundingMode",
  "GEN floating-point rounding mode",
  [
    I32EnumAttrCase<"UNUSED", 0, "unused">,
    I32EnumAttrCase<"RTE",    1, "rte">, ///< Round to nearest, ties to even
    I32EnumAttrCase<"RTN",    2, "rtn">, ///< Round toward negative
    I32EnumAttrCase<"RTP",    3, "rtp">, ///< Round toward positive
    I32EnumAttrCase<"RTZ",    4, "rtz">, ///< Round toward zero
  ]> {
  let cppNamespace = "::mlir::GEN";
}

/// Enum attribute of the different precision types.
def GEN_PrecisionTypeAttr : I32EnumAttr<"PrecisionType",
  "GEN precision type",
  [
    I32EnumAttrCase<"UNUSED", 0,  "unused">,
    I32EnumAttrCase<"U8",     1,  "u8">,
    I32EnumAttrCase<"U4",     2,  "u4">,
    I32EnumAttrCase<"U2",     3,  "u2">,
    I32EnumAttrCase<"S8",     4,  "s8">,
    I32EnumAttrCase<"S4",     5,  "s4">,
    I32EnumAttrCase<"S2",     6,  "s2">,
    I32EnumAttrCase<"BF8",    7,  "bf8">,
    I32EnumAttrCase<"TF32",   8,  "tf32">,
    I32EnumAttrCase<"BF16",   9,  "bf16">,
    I32EnumAttrCase<"FP16",   10, "fp16">
  ]> {
  let cppNamespace = "::mlir::GEN";
}

#endif // GEN_ATTRDEFS
