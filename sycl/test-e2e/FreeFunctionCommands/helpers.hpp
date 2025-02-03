#include <iostream>
#include <string>

static int Check(int *Data, int Expected, size_t Index, std::string TestName) {
  if (Data[Index] == Expected)
    return 0;
  std::cout << "Failed " << TestName << " at index " << Index << " : "
            << Data[Index] << " != " << Expected << std::endl;
  return 1;
}