//#include <iostream>

#include <windows.h>

#include "win_unload.hpp"



// working.
// need namespace, etc.

void preserve_lib(const std::string &PluginPath) {
  //std::cout << "preserve_lib: " << PluginPath <<  std::endl;
  volatile auto Result = (void *)LoadLibraryA(PluginPath.c_str());
}

