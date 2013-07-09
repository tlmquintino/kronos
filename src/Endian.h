#ifndef kronos_Endian_h
#define kronos_Endian_h

#include <algorithm>

namespace kronos {

//------------------------------------------------------------------------------------------

template <typename T>
T swap_endian(T u)
{
    union
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}

//------------------------------------------------------------------------------------------

template <class T>
void reverse_endian(T* u)
{
  unsigned char* m = reinterpret_cast<unsigned char*>(u);
  std::reverse(m, m + sizeof(T));
}

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
