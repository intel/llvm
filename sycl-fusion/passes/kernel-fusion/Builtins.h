#ifndef KERNEL_FUSION_PASS_SYCL_BUILTINS_H
#define KERNEL_FUSION_PASS_SYCL_BUILTINS_H

#include "Kernel.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
// Forward declarations
class AttributeList;
class Function;
class IRBuilderBase;
class LLVMContext;
class Module;
class Value;
template <typename T> class ArrayRef;
} // namespace llvm

namespace jit_compiler {
/// barrier builtin name
constexpr llvm::StringLiteral BarrierName{"_Z22__spirv_ControlBarrierjjj"};
/// get_global_size builtin name
constexpr llvm::StringLiteral GetGlobalSizeName{
    "_Z25__spirv_BuiltInGlobalSizei"};
/// get_group_id builtin name
constexpr llvm::StringLiteral GetGroupIDName{"_Z26__spirv_BuiltInWorkgroupIdi"};
// get_global_offset name
constexpr llvm::StringLiteral GetGlobalOffsetName{
    "_Z27__spirv_BuiltInGlobalOffseti"};
/// get_num_workgroups builtin name
constexpr llvm::StringLiteral GetNumWorkGroupsName{
    "_Z28__spirv_BuiltInNumWorkgroupsi"};
/// get_local_size builtin name
constexpr llvm::StringLiteral GetLocalSizeName{
    "_Z28__spirv_BuiltInWorkgroupSizei"};
/// get_global_linear_id builtin name
constexpr llvm::StringLiteral GetGlobalLinearIDName{
    "_Z29__spirv_BuiltInGlobalLinearIdv"};
/// get_local_id builtin name
constexpr llvm::StringLiteral GetLocalIDName{
    "_Z32__spirv_BuiltInLocalInvocationIdi"};
/// get_global_id builtin name
constexpr llvm::StringLiteral GetGlobalIDName{
    "_Z33__spirv_BuiltInGlobalInvocationIdi"};

///
/// @return The result of calling get_global_linear_id
///
/// @param Builder The builder.
/// @param FusedNDRange The range of the fused kernel.
llvm::Value *getGlobalLinearID(llvm::IRBuilderBase &Builder,
                               const NDRange &FusedNDRange);

///
/// Creates a call to a barrier function.
void barrierCall(llvm::IRBuilderBase &Builder, int Flags);

///
/// @return A call to a SPIRV function, which will be declared if not already in
/// the module.
llvm::Value *createSPIRVCall(llvm::IRBuilderBase &Builder, llvm::StringRef N,
                             llvm::ArrayRef<llvm::Value *> Args);

///
/// Remaps index space getters builtins.
llvm::Function *remapBuiltins(llvm::Function *F, const NDRange &SrcNDRange,
                              const NDRange &FusedNDRange);
} // namespace jit_compiler

#endif // KERNEL_FUSION_PASS_SYCL_BUILTINS_H
