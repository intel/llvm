#pragma once

#ifdef _WIN32
#include <string>

__declspec(dllexport) void *getPreloadedPlugin(const std::string &PluginPath);
#endif
