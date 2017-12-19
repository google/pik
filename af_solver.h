// Closed-form solutions for low-degree polynomial equations.

#ifndef AF_SOLVER_H_
#define AF_SOLVER_H_

#include <complex>

#include "compiler_specific.h"

namespace pik {

class Solver {
 public:
  using C = std::complex<double>;

  // Finds a complex-valued solution of the reduced quadratic equation
  // x^2 + bx + c = 0. x can be real or complex => T is double or C. Returns
  // the more accurate root (higher magnitude).
  template <typename T>
  static C SolveReducedQuadratic(const T b, const T c) {
    // Note: if discriminant is non-negative, can avoid a few extra mul/add and
    // one division by computing sqrt(double) instead.
    const auto sqrt_d = sqrt(C(b * b - 4.0 * c));
    // Choose the solution that is less affected by catastrophic cancellation.
    const auto pos = sqrt_d - b;
    const auto neg = -sqrt_d - b;
    const auto max_abs = std::abs(pos) > std::abs(neg) ? pos : neg;
    return max_abs * T(0.5);  // divide by 2*a
  }

  // Same as above, but returns both roots (potentially inaccurate for tiny c).
  template <typename T>
  static void SolveReducedQuadratic(const T b, const T c,
                                    C* PIK_RESTRICT roots) {
    const auto sqrt_d = sqrt(C(b * b - 4.0 * c));
    const double rcp_2a = 0.5;
    roots[0] = (sqrt_d - b) * rcp_2a;
    roots[1] = (-sqrt_d - b) * rcp_2a;
  }

  // Returns up to three not necessarily unique complex-valued solutions of
  // the depressed cubic equation t^3 + pt + q = 0.
  // [https://en.wikipedia.org/wiki/Cubic_function#Vieta.27s_substitution].
  // p, q are real and must not both be zero. Evaluating the equation at each
  // root is within 3E-6 of zero even for log10(p/q) = 13 or log10(q/p) = 13.
  static int SolveDepressedCubic(const double p, const double q,
                                 C* PIK_RESTRICT roots) {
    const C kCubeRoot1(-0.5, 0.8660254037844386);

    // Coefficients of quadratic equation [in w^3].
    const C w3_root = SolveReducedQuadratic(q, p * p * p / -27.0);
    // Both roots lead to the same result, so only use one.
    const auto w1 = pow(w3_root, 1.0 / 3);  // principal cube root
    const auto w2 = kCubeRoot1 * w1;
    const auto w3 = std::conj(kCubeRoot1) * w1;
    int num_roots = 0;
    num_roots = AddRoot(w1, p, roots, num_roots);
    num_roots = AddRoot(w2, p, roots, num_roots);
    num_roots = AddRoot(w3, p, roots, num_roots);
    return num_roots;
  }

 private:
  // Appends t to roots and returns num_roots + 1 (if it is safe to divide).
  static inline int AddRoot(const C w, const double p, C* PIK_RESTRICT roots,
                            const int num_roots) {
    if (std::abs(w) < 1E-14) return num_roots;
    const auto root = w - p / (w * 3.0);
    roots[num_roots] = root;
    return num_roots + 1;
  }
};

}  // namespace pik

#endif  // AF_SOLVER_H_
