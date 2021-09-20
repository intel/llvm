#include <Windows.h>
#include <stdio.h>

__declspec(dllexport) void touchLibraries() {}

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpReserved) {
    switch(fdwReason) {
        case DLL_PROCESS_ATTACH:
            LoadLibrary(L"igdrcl64.dll");
            LoadLibrary(L"intelocl64.dll");
            break;

        case DLL_THREAD_ATTACH:
            break;

        case DLL_THREAD_DETACH:
            break;

        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;
}
