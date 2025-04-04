#pragma once

#include <llvm/Support/Error.h>
#include <string>

namespace jit_compiler {

std::string formatError(llvm::Error &&Err, const std::string &Msg);

template <typename ResultType, typename... ExtraArgTypes>
static ResultType errorTo(llvm::Error &&Err, const std::string &Msg,
                          ExtraArgTypes... ExtraArgs) {
  // Cannot throw an exception here if LLVM itself is compiled without exception
  // support.
  return ResultType{formatError(std::move(Err), Msg).c_str(), ExtraArgs...};
}

} // namespace jit_compiler
