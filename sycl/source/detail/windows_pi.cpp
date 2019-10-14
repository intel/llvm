#include <windows.h>
#include <winreg.h>
#include <string>

void *loadOsLibrary(const std::string &PluginPath) {
  return (void *)LoadLibraryA(PluginPath.c_str());
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return GetProcAddress((HMODULE)Library, FunctionName.c_str());
}
