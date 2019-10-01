#include <dlfcn.h>

void *loadOsLibrary(const char *PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct.
  return dlopen(PluginPath, RTLD_NOW);
}

void *getOsLibraryFuncAddress(void *Library, const char *FunctionName) {
  return dlsym(Library, FunctionName);
}
