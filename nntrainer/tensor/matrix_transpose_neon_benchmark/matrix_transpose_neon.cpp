/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "./transpose_utils_neon.h"
#include "./transpose_utils.h"
#include <iostream>


// template <>
// void transpose_neon(
//     unsigned int M,
//     unsigned int N,
//     const float* src,
//     unsigned int ld_src,
//     float* dst,
//     unsigned int ld_dst) {
//   unsigned int ib = 0, jb = 0;
//   if (N % 8 > 0 && N % 8 < 4) {
//     // If the remainder has n < 4 columns, we use the SSE kernel for the
//     // remainder because it requires 2 * (2 * 4 + 2 * N) = 16 + 4N instructions
//     // instead of 3 * 8 + 2 * N = 24 + 2N instructions in the masked AVX2
//     // kernel.
//     for (ib = 0; ib + 8 <= M; ib += 8) {
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_8x8_neon(
//             &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       for (unsigned int i = ib; i < ib + 8; i += 4) {
//         transpose_kernel_mxn_neon_128<4>(
//             N - jb,
//             &src[i * ld_src + jb],
//             ld_src,
//             &dst[i + jb * ld_dst],
//             ld_dst);
//       }
//     }
//   } else if (N % 8 == 4) {
//     // If the remainder has 4 columns, we use the SSE kernel for the remainder
//     // because it requires 2 * 16 = 32 instructions instead of 3 * 8 + 2 * 4 =
//     // 32 instructions + looping overhead needed in the masked AVX2 kernel.
//     for (ib = 0; ib + 8 <= M; ib += 8) {
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_8x8_neon(
//             &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       for (unsigned int i = ib; i < ib + 8; i += 4) {
//         transpose_kernel_4x4_neon(
//             &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
//       }
//     }
//   } else {
//     for (ib = 0; ib + 8 <= M; ib += 8) {
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_8x8_neon(
//             &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_256<8>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//     }
//   }

//   // Specialization for small M - ib cases so that the compiler can inline
//   // transpose_kernel_mxn_neon and unroll the loops whose iteration count
//   // depends on by M - ib .
//   // Specialization for m helps more than for n in transpose_kernel_mxn_neon
//   // because we have more loops in that function whose iteration count depends
//   // on m.
//   switch (M - ib) {
//     case 1:
//       for (unsigned int j = 0; j < N; ++j) {
//         dst[ib + j * ld_dst] = src[ib * ld_src + j];
//       }
//       break;
//     case 2:
//       for (jb = 0; jb + 4 <= N; jb += 4) {
//         transpose_kernel_mxn_neon_128<2>(
//             4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_128<2>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//     case 3:
//       for (jb = 0; jb + 4 <= N; jb += 4) {
//         transpose_kernel_mxn_neon_128<3>(
//             4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_128<3>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//     case 4:
//       for (jb = 0; jb + 4 <= N; jb += 4) {
//         transpose_kernel_4x4_neon(
//             &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_128<4>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//     case 5:
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_mxn_neon_256<5>(
//             8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_256<5>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//     case 6:
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_mxn_neon_256<6>(
//             8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_256<6>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//     case 7:
//       for (jb = 0; jb + 8 <= N; jb += 8) {
//         transpose_kernel_mxn_neon_256<7>(
//             8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
//       }
//       if (jb < N) {
//         transpose_kernel_mxn_neon_256<7>(
//             N - jb,
//             &src[ib * ld_src + jb],
//             ld_src,
//             &dst[ib + jb * ld_dst],
//             ld_dst);
//       }
//       break;
//   }
// }

template <>
void transpose_neon(
    unsigned int M,
    unsigned int N,
    const __fp16* src,
    unsigned int ld_src,
    __fp16* dst,
    unsigned int ld_dst) {
  unsigned int ib = 0, jb = 0;
  if (N % 8 > 0 && N % 8 < 4) {
    // If the remainder has n < 4 columns, we use the SSE kernel for the
    // remainder because it requires 2 * (2 * 4 + 2 * N) = 16 + 4N instructions
    // instead of 3 * 8 + 2 * N = 24 + 2N instructions in the masked AVX2
    // kernel.
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        // std::cout << "if : transpose_kernel_8x8_neon\n";
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (unsigned int i = ib; i < ib + 8; i += 4) {
        std::cout << "if : transpose_kernel_mxn_neon_128<4>\n";
        transpose_kernel_mxn_neon_128<4>(
            N - jb,
            &src[i * ld_src + jb],
            ld_src,
            &dst[i + jb * ld_dst],
            ld_dst);
      }
    }
  } else if (N % 8 == 4) {
    // If the remainder has 4 columns, we use the SSE kernel for the remainder
    // because it requires 2 * 16 = 32 instructions instead of 3 * 8 + 2 * 4 =
    // 32 instructions + looping overhead needed in the masked AVX2 kernel.
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        // std::cout << "else if == 4: transpose_kernel_8x8_neon\n";
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      for (unsigned int i = ib; i < ib + 8; i += 4) {
        std::cout << "else if == 4: transpose_kernel_4x4_neon\n";
        transpose_kernel_4x4_neon(
            &src[i * ld_src + jb], ld_src, &dst[i + jb * ld_dst], ld_dst);
      }
    }
  } else {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 8 <= N; jb += 8) {
        // std::cout << "else : transpose_kernel_8x8_neon\n";
        transpose_kernel_8x8_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      // std::cout << "jb : " << jb << std::endl;
      if (jb < N) {
        // std::cout << "transpose_kernel_mxn_neon_256<8>\n";
        transpose_kernel_mxn_neon_256<8>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
    }
  }

  // Specialization for small M - ib cases so that the compiler can inline
  // transpose_kernel_mxn_neon and unroll the loops whose iteration count
  // depends on by M - ib .
  // Specialization for m helps more than for n in transpose_kernel_mxn_neon
  // because we have more loops in that function whose iteration count depends
  // on m.
  // std::cout << "ib : " << ib << std::endl;
  switch (M - ib) {
    case 1:
      for (unsigned int j = 0; j < N; ++j) {
        dst[ib + j * ld_dst] = src[ib * ld_src + j];
      }
      break;
    case 2:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_mxn_neon_128<2>(
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<2>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 3:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_mxn_neon_128<3>(
            4, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<3>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 4:
      for (jb = 0; jb + 4 <= N; jb += 4) {
        transpose_kernel_4x4_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_128<4>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 5:
        std::cout << "transpose_kernel_mxn_neon_256<5>\n";
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<5>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<5>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 6:
        std::cout << "transpose_kernel_mxn_neon_256<6>\n";
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<6>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<6>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
    case 7:
        std::cout << "transpose_kernel_mxn_neon_256<7>\n";
      for (jb = 0; jb + 8 <= N; jb += 8) {
        transpose_kernel_mxn_neon_256<7>(
            8, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N) {
        transpose_kernel_mxn_neon_256<7>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
      break;
  }
}

#ifdef ENABLE_AVX
template <>
void transpose_neon(
    unsigned int M,
    unsigned int N,
    const uint8_t* src,
    unsigned int ld_src,
    uint8_t* dst,
    unsigned int ld_dst) {
  unsigned int ib = 0, jb = 0;
  if (M >= 8) {
    for (ib = 0; ib + 8 <= M; ib += 8) {
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_8x32_neon(
            &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }

      if (jb < N) {
        transpose_kernel_mxn_neon_uint8<8>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      }
    }
  }

  // Specialization for small M - ib cases
  switch (M - ib) {
    case 1:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<1>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }

      if (jb < N)
        transpose_kernel_mxn_neon_uint8<1>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);

      break;
    case 2:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<2>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<2>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 3:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<3>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<3>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 4:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<4>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<4>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 5:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<5>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<5>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 6:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<6>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<6>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
    case 7:
      for (jb = 0; jb + 32 <= N; jb += 32) {
        transpose_kernel_mxn_neon_uint8<7>(
            32, &src[ib * ld_src + jb], ld_src, &dst[ib + jb * ld_dst], ld_dst);
      }
      if (jb < N)
        transpose_kernel_mxn_neon_uint8<7>(
            N - jb,
            &src[ib * ld_src + jb],
            ld_src,
            &dst[ib + jb * ld_dst],
            ld_dst);
      break;
  }
}

template <>
void transpose_neon(
    unsigned int M,
    unsigned int N,
    const uint16_t* src,
    unsigned int ld_src,
    uint16_t* dst,
    unsigned int ld_dst) {
  unsigned int i = 0;
  for (; i < M / 8 * 8; i += 8) {
    unsigned int j = 0;
    for (; j < N / 16 * 16; j += 16) {
      transpose_kernel_8x16_neon<false, false>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst);
    }
    // handle j rem
    unsigned nrem = N - j;
    if (nrem > 0) {
      transpose_kernel_8x16_neon<false, true>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, 8, nrem);
    }
  }

  // handle i rem
  unsigned mrem = M - i;
  if (mrem > 0) {
    unsigned int j = 0;
    for (; j < N / 16 * 16; j += 16) {
      transpose_kernel_8x16_neon<true, false>(
          src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, mrem, 16);
    }
    // handle j rem
    unsigned nrem = N - j;
    transpose_kernel_8x16_neon<true, true>(
        src + i * ld_src + j, ld_src, dst + j * ld_dst + i, ld_dst, mrem, nrem);
  }
}

#endif

