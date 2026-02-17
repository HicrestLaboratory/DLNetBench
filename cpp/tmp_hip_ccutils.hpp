#ifndef __TMP_HIP_CCUTILS_HPP__
#define __TMP_HIP_CCUTILS_HPP__

#include <hip/hip_runtime.h> // HIP

#define CCUTILS_HIP_FREE_SAFE(data)         \
    do {                                    \
      if ((data) != nullptr) hipFree(data); \
    } while (0)


#define CCUTILS_HIP_CHECK(call) {                                               \
  hipError_t err = call;                                                        \
  if (err != hipSuccess) {                                                      \
    fprintf(stderr, "HIP error in file '%s' in line %i : %s (%u)\n",            \
            __FILE__, __LINE__, hipGetErrorString(err), err);                   \
    exit(err);                                                                  \
  }                                                                             \
}


#endif // __TMP_HIP_CCUTILS_HPP__

