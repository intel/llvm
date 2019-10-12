#include <dlfcn.h>
#include <string>

void *loadOsLibrary(const std::string &PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct.
  return dlopen(PluginPath.c_str(), RTLD_NOW);
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return dlsym(Library, FunctionName.c_str());
}
