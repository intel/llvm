#include <iostream>

#include <windows.h>

#include "win_unload.hpp"



// working.
// need namespace, etc.

static void* oclPtr = nullptr;
static void* l0Ptr  = nullptr;

__declspec(dllexport) void* preserve_lib(const std::string &PluginPath) {
  std::cout << "preserve_lib: " << PluginPath <<  std::endl;
  if(PluginPath.find("opencl.dll") != std::string::npos){
    return oclPtr;
  }
  if(PluginPath.find("level_zero.dll") != std::string::npos){
    return l0Ptr;
  }
  return nullptr;
  //void*  Result = (void *)LoadLibraryA(PluginPath.c_str());
  //return Result;
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, // handle to DLL module
                    DWORD fdwReason,    // reason for calling function
                    LPVOID lpReserved)  // reserved
{
  //TCHAR dllFilePath[512 + 1] = { 0 };
  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    //GetModuleFileNameA(hinstDLL, dllFilePath, 512);
    //printf(">> Module   load: %s\n", dllFilePath);
    std::cout << "win_unload process_attach" << std::endl;
    oclPtr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_opencl.dll");
    l0Ptr = LoadLibraryA("C:\\iusers\\cperkins\\sycl_workspace\\build\\bin\\pi_level_zero.dll");
    break;
  case DLL_PROCESS_DETACH:
    //GetModuleFileNameA(hinstDLL, dllFilePath, 512);
    //printf(">> Module Unload: %s\n", dllFilePath);
    std::cout << "win_unload  process_detach" << std::endl;
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}


