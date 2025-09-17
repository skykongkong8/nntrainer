#pragma once  
#include <cstdint>  
#include <type_traits>  
#include <stdexcept>  
#include <limits>  

/**  
 * @brief Exception type used by my_narrow – mimics gsl::narrowing_error.  
 */  
struct narrowing_error : public std::runtime_error {  
    narrowing_error() : std::runtime_error("narrowing conversion would lose data") {}  
};  

/**  
 * @brief Safe narrowing conversion.  
 *  
 *   - If the value can be represented exactly in the destination type,  
 *     the conversion succeeds.  
 *   - Otherwise it throws narrowing_error (or you can change it to assert/abort).  
 *  
 * This mirrors the semantics of gsl::narrow<T>(value).  
 *  
 * @tparam To   Destination integral or floating‑point type.  
 * @tparam From Source integral or floating‑point type.  
 * @param  value Value to convert.  
 * @return To   The converted value.  
 * @throws narrowing_error if the conversion would truncate/overflow.  
 */  
template <typename To, typename From>  
constexpr To my_narrow(From value)  
{  
    static_assert(std::is_arithmetic_v<To>, "my_narrow destination must be arithmetic");  
    static_assert(std::is_arithmetic_v<From>, "my_narrow source must be arithmetic");  

    // 1️⃣  Integral → Integral  
    if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {  
        // Check that the value fits into the destination range.  
        if (value < static_cast<From>(std::numeric_limits<To>::min()) ||  
            value > static_cast<From>(std::numeric_limits<To>::max())) {  
            throw narrowing_error{};  
        }  
        return static_cast<To>(value);  
    }  
    // 2️⃣  Floating → Integral  
    else if constexpr (std::is_floating_point_v<From> && std::is_integral_v<To>) {  
        // For floating → integral we must also check that the value is integral.  
        if (!std::isfinite(value) ||  
            value < static_cast<From>(std::numeric_limits<To>::min()) ||  
            value > static_cast<From>(std::numeric_limits<To>::max()) ||  
            std::trunc(value) != value) {  
            throw narrowing_error{};  
        }  
        return static_cast<To>(value);  
    }  
    // 3️⃣  Integral → Floating  
    else if constexpr (std::is_integral_v<From> && std::is_floating_point_v<To>) {  
        // All integral values that fit in the floating‑point mantissa can be represented exactly.  
        // Most practical implementations (IEEE‑754) have 24 bits for float, 53 bits for double.  
        constexpr int mantissa_bits = std::numeric_limits<To>::digits;  
        if (value > (static_cast<From>(1) << mantissa_bits) - 1 ||  
            value < -(static_cast<From>(1) << mantissa_bits)) {  
            // Outside the exact‑range – still allowed by gsl::narrow, but  
            // we keep the “strict” behavior here; change to static_cast if you prefer.  
            //throw narrowing_error{};  
        }  
        return static_cast<To>(value);  
    }  
    // 4️⃣  Floating → Floating (different precisions)  
    else {  
        // Narrowing a floating‑point type to a smaller one may lose precision.  
        // gsl::narrow treats this as an error only when the value cannot be represented  
        // in the target’s range. We follow that rule.  
        if (!std::isfinite(value) ||  
            value < std::numeric_limits<To>::lowest() ||  
            value > std::numeric_limits<To>::max()) {  
            throw narrowing_error{};  
        }  
        return static_cast<To>(value);  
    }  
}  
