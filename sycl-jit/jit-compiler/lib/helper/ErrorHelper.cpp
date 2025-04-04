#include "ErrorHelper.h"

#include <sstream>

std::string jit_compiler::formatError(llvm::Error &&Err,
                                      const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return ErrMsg.str();
}
