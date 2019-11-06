#include <windows.h>
#include <winreg.h>
#include <string>

namespace cl {
namespace sycl {
namespace detail {
namespace pi {

void *loadOsLibrary(const std::string &PluginPath) {
  return (void *)LoadLibraryA(PluginPath.c_str());
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return GetProcAddress((HMODULE)Library, FunctionName.c_str());
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // namespace cl
