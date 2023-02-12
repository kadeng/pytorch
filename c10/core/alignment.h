#pragma once

#include <cstddef>
#ifdef __linux__
#include <unistd.h>
#endif

namespace c10 {

#ifdef C10_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
constexpr size_t gAlignment = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;
#endif

#ifdef __linux__
// inorder to enable thp, buffers need to be page aligned
const long gPagesize = sysconf(_SC_PAGESIZE);
// for kernels that don't provide page size, default it to 4K
const size_t gAlignment_thp = (gPagesize < 0 ? 4096 : gPagesize);
#endif

} // namespace c10
