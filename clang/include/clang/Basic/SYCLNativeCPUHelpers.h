#include "clang/Basic/LangOptions.h"
#include <string>
namespace clang {
inline std::string getNativeCPUHeaderName(const LangOptions &LangOpts) {
  return LangOpts.SYCLIntHeader + ".hc";
}
} // namespace clang
