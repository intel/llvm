#if defined(_WIN32)
#ifdef EXPORT
__declspec(dllexport)
#else
__declspec(dllimport)
#endif
#endif
int wrapper();
