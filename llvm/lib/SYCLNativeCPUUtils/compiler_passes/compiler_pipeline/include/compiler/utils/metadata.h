// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMPILER_UTILS_METADATA_H_INCLUDED
#define COMPILER_UTILS_METADATA_H_INCLUDED

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Metadata.h>

#include <optional>

#include "vectorization_factor.h"

namespace llvm {
class Function;
class Module;
}  // namespace llvm

namespace compiler {
namespace utils {

/// @brief OpenCL C standard to target.
enum OpenCLCVer {
  /// @brief OpenCL C 1.0
  OpenCLC10 = (1 * 100 + 0) * 1000,
  /// @brief OpenCL C 1.1
  OpenCLC11 = (1 * 100 + 1) * 1000,
  /// @brief OpenCL C 1.2
  OpenCLC12 = (1 * 100 + 2) * 1000,
  /// @brief OpenCL C 2.0
  OpenCLC20 = (2 * 100 + 0) * 1000,
  /// @brief OpenCL C 3.0
  OpenCLC30 = (3 * 100 + 0) * 1000,
};

/// @brief Returns the OpenCL version, encoded as (Major*100 + Minor)*1000.
///
/// If the Module does not contain any information, then OpenCLC12 is returned.
uint32_t getOpenCLVersion(const llvm::Module &m);

/// @brief Describes the state of vectorization on a function/loop.
struct VectorizationInfo {
  /// @brief The VectorizationFactor. A scalar value if unvectorized.
  VectorizationFactor vf;
  /// @brief The dimension along which vectorization took place.
  unsigned simdDimIdx;
  /// @brief Whether or not the function/loop was vector-predicated.
  bool IsVectorPredicated;
};

/// @brief Encodes metadata indicating vectorization failure to a kernel, along
/// with the the vectorization factor and dimension that failed.
///
/// @param[in] f Function in which to encode the link.
/// @param[in] info Vectorization info serving as the key.
void encodeVectorizationFailedMetadata(llvm::Function &f,
                                       const VectorizationInfo &info);

/// @brief Encodes the vectorization metadata linking the original kernel to a
/// vectorized one, using the vectorization factor and dimension as the key.
///
/// @param[in] origF Original function in which to encode the link.
/// @param[in] vectorizedF Vectorized function to link.
/// @param[in] info Vectorization factor serving as the key.
void linkOrigToVeczFnMetadata(llvm::Function &origF,
                              llvm::Function &vectorizedF,
                              const VectorizationInfo &info);

/// @brief Encodes the vectorization metadata linking a vectorized kernel back
/// to its original one, using the vectorization factor and dimension as the
/// key.
///
/// @param[in] vectorizedF Vectorized function in which to encode the link.
/// @param[in] origF Original function to link.
/// @param[in] info Vectorization factor serving as the key.
void linkVeczToOrigFnMetadata(llvm::Function &vectorizedF,
                              llvm::Function &origF,
                              const VectorizationInfo &info);

using LinkMetadataResult = std::pair<llvm::Function *, VectorizationInfo>;

/// @brief Decodes the metadata linking a kernel to its vectorized variant.
///
/// @param[in] f Function for which to decode the metadata.
/// @param[out] factors unordered vector of recovered vectorization links.
///
/// @return true on success, false if there is no vectorization metadata for the
/// function.
bool parseOrigToVeczFnLinkMetadata(
    llvm::Function &f, llvm::SmallVectorImpl<LinkMetadataResult> &factors);

/// @brief Decodes the metadata linking a vectorized kernel back to its
/// original one.
///
/// @param[in] f Function for which to decode the metadata.
///
/// @return On success, a pair containing a pointer to the original kernel
/// function and the vectorization factor used as the key. The original
/// function may be null. On decoding failure, std::nullopt.
std::optional<LinkMetadataResult> parseVeczToOrigFnLinkMetadata(
    llvm::Function &f);

/// @brief Drops "base" vectorization metadata from a function, if present.
///
/// @param[in] f Function to drop metadata from.
void dropVeczOrigMetadata(llvm::Function &f);

/// @brief Drops "derived" vectorization metadata from a function, if present.
///
/// @param[in] f Function to drop metadata from.
void dropVeczDerivedMetadata(llvm::Function &f);

/// @brief Encodes metadata indicating the various components that constitute a
/// kernel function wrapped with the WorkItemLoopsPass.
///
/// @param[in] f Function in which to encode the metadata.
/// @param[in] simdDimIdx The dimension (0,1,2) along which vectorization took
/// place.
/// @param[in] mainInfo VectorizationInfo used on the 'main' work-item
/// iterations.
/// @param[in] tailInfo VectorizationInfo used on the tail iterations, if
/// applicable.
///
/// Note that a 'tail' is defined as the work done to execute work-items not
/// covered by the 'main' body. Therefore an unvectorized kernel should expect
/// a scalar 'main' vectorization factor and no 'tail' (rather than the other
/// way round).

/// Some examples of *typical* usage:
/// 1. An unvectorized kernel will encode a scalar VF for the main iterations
/// and nothing for the tail ones.
/// 2. A vectorized kernel will encode vectorization factor for its main
/// iterations. If it handles the case in which the local work-group size does
/// not evenly divide the vectorization factor, it will encode how it manages
/// the tail iterations. This is *typically* with a series of scalar
/// iterations, encoded in tailVF.
/// 3. Vector-predicated kernels with no tails will encode the *maximum* VF used
/// for the main loop, with no tail iterations.
///
/// This metadata is encoded as:
/// define void @foo() !codeplay_ca_wrapper !X
/// !X = { !Main, !Tail }
/// !Main = { i32 mKnownMin, i32 mIsScalable, i32 simdDimIdx, i32 mIsVP }
/// if tailVF is None:
///   !Tail = {}
/// else
///   !Tail = { i32 tKnownMin, i32 tIsScalable, i32 simdDimIdx, i32 tIsVP }
void encodeWrapperFnMetadata(llvm::Function &f,
                             const VectorizationInfo &mainInfo,
                             std::optional<VectorizationInfo> tailInfo);

/// @brief Decodes the metadata describing a wrapped kernel's loop structure.
///
/// @param[in] f Function for which to decode the metadata.
///
/// @return On success, a pair containing the VectorizationInfo for the main
/// loop(s) and the (optional) VectorizationInfo info for the tail loop(s). On
/// decoding failure, std::nullopt.
std::optional<std::pair<VectorizationInfo, std::optional<VectorizationInfo>>>
parseWrapperFnMetadata(llvm::Function &f);

/// @brief Copies function metadata from one function to another.
///
/// @param[in] fromF Function from which to copy the metadata.
/// @param[in] toF Function onto which to copy the metadata.
/// @param[in] includeDebug Whether or not to copy debug function metadata.
void copyFunctionMetadata(llvm::Function &fromF, llvm::Function &toF,
                          bool includeDebug = false);

/// @brief Encodes information about a function's local work group size as
/// metadata.
///
/// @param[in] f Function in which to encode the metadata.
/// @param[in] localSizes array of size information to encode.
void encodeLocalSizeMetadata(llvm::Function &f,
                             const std::array<uint64_t, 3> &localSizes);

/// @brief Retrieves information about a function's local sizes via metadata.
///
/// @param[in] f Function from which to decode the metadata
/// @returns The local size array if present, else `std::nullopt`
std::optional<std::array<uint64_t, 3>> getLocalSizeMetadata(
    const llvm::Function &f);

/// @brief Drops all !mux_scheduled_fn metadata from a function.
void dropSchedulingParameterMetadata(llvm::Function &f);

/// @brief Retrieves the indices of scheduling parameters from the function.
llvm::SmallVector<int, 4> getSchedulingParameterFunctionMetadata(
    const llvm::Function &f);

/// @brief Sets scheduling-parameter metadata on the given function
void setSchedulingParameterFunctionMetadata(llvm::Function &f,
                                            llvm::ArrayRef<int> idxs);

/// @brief Sets module-level metadata describing the set of scheduling
/// parameters.
void setSchedulingParameterModuleMetadata(llvm::Module &m,
                                          llvm::ArrayRef<std::string> names);

/// @brief Retrieves module-level metadata describing the set of scheduling
/// parameters or nullptr.
llvm::NamedMDNode *getSchedulingParameterModuleMetadata(const llvm::Module &m);

/// @brief If the given function parameter index is considered a scheduling
/// parameter, it returns the corresponding index into the target's list of
/// scheduling parameters.
///
/// It uses !mux_scheduled_fn metadata for this check.
std::optional<unsigned> isSchedulingParameter(const llvm::Function &f,
                                              unsigned idx);

/// @brief Extracts the required work group size from a kernel's function
/// metadata.
///
/// @param[in] f Kernel for extraction.
///
/// @return The work group size or std::nullopt if there is no such metadata.
std::optional<std::array<uint64_t, 3>> parseRequiredWGSMetadata(
    const llvm::Function &f);

/// @brief Extracts the required work group size from an opencl.kernels subnode,
/// which is similar to the function metadata, but the size is stored under
/// different indices than on a function.
///
/// @param[in] node Kernel's subnode for extraction.
///
/// @return The work group size or std::nullopt if there is no such metadata.
std::optional<std::array<uint64_t, 3>> parseRequiredWGSMetadata(
    const llvm::MDNode &node);

/// @brief Extracts the maximum work dimension from a kernel's function
/// metadata
///
/// @param[in] f Kernel for extraction.
///
/// @return The maximum work dimension or std::nullopt if there is no such
/// metadata.
std::optional<uint32_t> parseMaxWorkDimMetadata(const llvm::Function &f);

/// @brief Describes the state of vectorization on a function/loop.
struct KernelInfo {
  explicit KernelInfo(llvm::StringRef name) : Name(name) {}
  /// @brief The function name
  std::string Name;
  /// @brief The required work-group size. Optional.
  std::optional<std::array<uint64_t, 3>> ReqdWGSize;
};

/// @brief Helper function to populate a list of kernels and associated
/// information from a module.
///
/// @param m Module to retrieve kernels from
/// @param results List of kernel info parsed from metadata or taken from the
/// module.
void populateKernelList(llvm::Module &m,
                        llvm::SmallVectorImpl<KernelInfo> &results);

/// @brief Replaces instances of kernel fromF with toF in module-level
/// !opencl.kernels metadata.
/// @param fromF Function to replace with toF in metadata
/// @param toF Function with which to replace references to fromF
/// @param M Module in which to find the metadata
void replaceKernelInOpenCLKernelsMetadata(llvm::Function &fromF,
                                          llvm::Function &toF, llvm::Module &M);

/// @brief Encodes information about a function's local work group size as
/// metadata.
///
/// @param[in] f Function in which to encode the metadata.
/// @param[in] size sub-group size information to encode.
void encodeReqdSubgroupSizeMetadata(llvm::Function &f, uint32_t size);

/// @brief Retrieves information about a function's required sub-group size via
/// metadata.
///
/// @param[in] f Function from which to decode the metadata
/// @returns The required sub-group size if present, else `std::nullopt`
std::optional<uint32_t> getReqdSubgroupSize(const llvm::Function &f);

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_METADATA_H_INCLUDED
