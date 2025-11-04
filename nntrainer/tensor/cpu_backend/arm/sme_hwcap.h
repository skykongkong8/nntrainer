// sme_hwcap.h
#pragma once
#include <stdbool.h>
#include <sys/auxv.h>

// Some NDK headers might not have HWCAP2_*, so perhaps do:
#ifndef HWCAP2_SME
#define HWCAP2_SME   (1UL << 23)  // Use aligned with kernel definition per-Platforms
#endif
#ifndef HWCAP2_SME2
#define HWCAP2_SME2  (1UL << 24)
#endif

static inline bool has_sme() {
    unsigned long hw2 = getauxval(AT_HWCAP2);
    return (hw2 & HWCAP2_SME) != 0;
}
static inline bool has_sme2() {
    unsigned long hw2 = getauxval(AT_HWCAP2);
    return (hw2 & HWCAP2_SME2) != 0;
}
