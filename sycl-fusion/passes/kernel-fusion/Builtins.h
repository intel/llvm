#ifndef KERNEL_FUSION_PASS_SYCL_BUILTINS_H
#define KERNEL_FUSION_PASS_SYCL_BUILTINS_H

#include "llvm/Support/Error.h"

#include <map>

namespace llvm {
// Forward declarations
class Function;
class IRBuilderBase;
class Value;
class TargetFusionInfo;
} // namespace llvm

namespace jit_compiler {
// Forward declaration
class NDRange;

///
/// @return The result of calling get_global_linear_id
///
/// @param Builder The builder.
/// @param TargetInfo The object providing target-specific abstractions.
/// @param FusedNDRange The range of the fused kernel.
llvm::Value *getGlobalLinearID(llvm::IRBuilderBase &Builder,
                               const llvm::TargetFusionInfo &TargetInfo,
                               const NDRange &FusedNDRange);

///
/// Enumeration of index space getters that may need to be remapped in a fused
/// kernel.
enum class BuiltinKind : uint8_t {
  GlobalSizeRemapper,
  LocalSizeRemapper,
  NumWorkGroupsRemapper,
  GlobalOffsetRemapper,
  GlobalIDRemapper,
  LocalIDRemapper,
  GroupIDRemapper,
};

///
/// Implements the target-independent parts of the the remapping approach.
class Remapper {
public:
  explicit Remapper(const llvm::TargetFusionInfo &TargetInfo)
      : TargetInfo(TargetInfo) {}

  ///
  /// Generate a unique function name for a remapper function.
  static std::string getFunctionName(BuiltinKind K, const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange,
                                     uint32_t Idx = -1);

  ///
  /// Recursively remap index space getters builtins.
  llvm::Expected<llvm::Function *> remapBuiltins(llvm::Function *F,
                                                 const NDRange &SrcNDRange,
                                                 const NDRange &FusedNDRange);

  ///
  /// Recompute the original (i.e. under \p SrcNDRange) value of \p K  in the
  /// context of the \p FusedNDRange. The \p Index corresponds selects the SYCL
  /// dimension.
  llvm::Value *remap(BuiltinKind K, llvm::IRBuilderBase &Builder,
                     const NDRange &SrcNDRange, const NDRange &FusedNDRange,
                     uint32_t Index) const;

  ///
  /// @return the default value for builtin \p K, i.e. 0 for IDs and 1 for
  /// sizes.
  unsigned getDefaultValue(BuiltinKind K) const;

private:
  std::map<std::tuple<llvm::Function *, const NDRange &, const NDRange &>,
           llvm::Function *>
      Cache;
  const llvm::TargetFusionInfo &TargetInfo;
};
} // namespace jit_compiler

#endif // KERNEL_FUSION_PASS_SYCL_BUILTINS_H
