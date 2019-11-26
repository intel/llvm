#include <dlfcn.h>
#include <string>

namespace cl {
namespace sycl {
namespace detail {
namespace pi {

void *loadOsLibrary(const std::string &PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct. Explore using
  // RTLD_DEEPBIND option when there are multiple plugins.
  return dlopen(PluginPath.c_str(), RTLD_NOW);
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}

} // namespace pi
} // namespace detail
} // namespace sycl
} // namespace cl
