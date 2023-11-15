//==----- KernelIO.h - YAML output of internal SYCL kernel representation --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_KERNELIO_H
#define SYCL_FUSION_COMMON_KERNELIO_H

#include "Kernel.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

// Specify how to map std::vectors of different user-defined types to YAML
// sequences.
LLVM_YAML_IS_SEQUENCE_VECTOR(jit_compiler::ArgUsageMask)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(jit_compiler::ParameterKind)
LLVM_YAML_IS_SEQUENCE_VECTOR(jit_compiler::SYCLArgumentDescriptor)
LLVM_YAML_IS_SEQUENCE_VECTOR(jit_compiler::SYCLKernelAttribute)
LLVM_YAML_IS_SEQUENCE_VECTOR(jit_compiler::SYCLKernelInfo)

//
// Mapping traits for the different elements of KernelInfo.
namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<jit_compiler::ParameterKind> {
  static void enumeration(IO &IO, jit_compiler::ParameterKind &PK) {
    IO.enumCase(PK, "Accessor", jit_compiler::ParameterKind::Accessor);
    IO.enumCase(PK, "StdLayout", jit_compiler::ParameterKind::StdLayout);
    IO.enumCase(PK, "Sampler", jit_compiler::ParameterKind::Sampler);
    IO.enumCase(PK, "Pointer", jit_compiler::ParameterKind::Pointer);
    IO.enumCase(PK, "SpecConstantBuffer",
                jit_compiler::ParameterKind::SpecConstBuffer);
    IO.enumCase(PK, "Stream", jit_compiler::ParameterKind::Stream);
    IO.enumCase(PK, "Invalid", jit_compiler::ParameterKind::Invalid);
  }
};

template <> struct ScalarEnumerationTraits<jit_compiler::BinaryFormat> {
  static void enumeration(IO &IO, jit_compiler::BinaryFormat &BF) {
    IO.enumCase(BF, "LLVM", jit_compiler::BinaryFormat::LLVM);
    IO.enumCase(BF, "SPIRV", jit_compiler::BinaryFormat::SPIRV);
    IO.enumCase(BF, "PTX", jit_compiler::BinaryFormat::PTX);
    IO.enumCase(BF, "AMDGCN", jit_compiler::BinaryFormat::AMDGCN);
    IO.enumCase(BF, "INVALID", jit_compiler::BinaryFormat::INVALID);
  }
};

template <> struct MappingTraits<jit_compiler::SYCLKernelBinaryInfo> {
  static void mapping(IO &IO, jit_compiler::SYCLKernelBinaryInfo &BI) {
    IO.mapRequired("Format", BI.Format);
    IO.mapRequired("AddressBits", BI.AddressBits);
    // We do not serialize the pointer here on purpose.
    IO.mapRequired("BinarySize", BI.BinarySize);
  }
};

template <> struct MappingTraits<jit_compiler::SYCLArgumentDescriptor> {
  static void mapping(IO &IO, jit_compiler::SYCLArgumentDescriptor &AD) {
    IO.mapRequired("Kinds", AD.Kinds);
    IO.mapRequired("Mask", AD.UsageMask);
  }
};

template <> struct MappingTraits<jit_compiler::SYCLKernelAttribute> {
  static void mapping(IO &IO, jit_compiler::SYCLKernelAttribute &KA) {
    IO.mapRequired("AttrName", KA.AttributeName);
    IO.mapRequired("Values", KA.Values);
  }
};

template <> struct MappingTraits<jit_compiler::SYCLKernelInfo> {
  static void mapping(IO &IO, jit_compiler::SYCLKernelInfo &KI) {
    IO.mapRequired("KernelName", KI.Name);
    IO.mapRequired("Args", KI.Args);
    IO.mapOptional("Attributes", KI.Attributes);
    IO.mapRequired("BinInfo", KI.BinaryInfo);
  }
};

template <> struct MappingTraits<jit_compiler::SYCLModuleInfo> {
  static void mapping(IO &IO, jit_compiler::SYCLModuleInfo &SMI) {
    IO.mapRequired("Kernels", SMI.kernels());
  }
};

} // namespace yaml
} // namespace llvm

#endif // SYCL_FUSION_COMMON_KERNELIO_H
