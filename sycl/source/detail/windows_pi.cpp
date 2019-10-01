#include <windows.h>
#include <winreg.h>

void *loadOsLibrary(const char *PluginPath) {
  // TODO: Check if the option RTLD_NOW is correct.
  return (void *)LoadLibraryA(PluginPath);
}

void *getOsLibraryFuncAddress(void *Library, const char *FunctionName) {
  return GetProcAddress((HMODULE)Library, FunctionName);
}
