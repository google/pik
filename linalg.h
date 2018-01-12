#ifndef LINALG_H_
#define LINALG_H_

#include <stddef.h>
#include <cmath>
#include <vector>

#include "image.h"
#include "status.h"

namespace pik {

using ImageD = Image<double>;

inline double DotProduct(const size_t N,
                         const double* const PIK_RESTRICT a,
                         const double* const PIK_RESTRICT b) {
  double sum = 0.0;
  for (int k = 0; k < N; ++k) {
    sum += a[k] * b[k];
  }
  return sum;
}

inline double DotProduct(const ImageD& a, const ImageD& b) {
  PIK_ASSERT(a.ysize() == 1);
  PIK_ASSERT(b.ysize() == 1);
  PIK_ASSERT(a.xsize() == b.xsize());
  const double* const PIK_RESTRICT row_a = a.Row(0);
  const double* const PIK_RESTRICT row_b = b.Row(0);
  return DotProduct(a.xsize(), row_a, row_b);
}

inline ImageD Transpose(const ImageD& A) {
  ImageD out(A.ysize(), A.xsize());
  for (size_t x = 0; x < A.xsize(); ++x) {
    double* const PIK_RESTRICT row_out = out.Row(x);
    for (size_t y = 0; y < A.ysize(); ++y) {
      row_out[y] = A.Row(y)[x];
    }
  }
  return out;
}

template<typename Tout, typename Tin1, typename Tin2>
Image<Tout> MatMul(const Image<Tin1>& A, const Image<Tin2>& B) {
  PIK_ASSERT(A.ysize() == B.xsize());
  Image<Tout> out(A.xsize(), B.ysize());
  for (size_t y = 0; y < B.ysize(); ++y) {
    const Tin2* const PIK_RESTRICT row_b = B.Row(y);
    Tout* const PIK_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < A.xsize(); ++x) {
      row_out[x] = 0.0;
      for (size_t k = 0; k < B.xsize(); ++k) {
        row_out[x] += A.Row(k)[x] * row_b[k];
      }
    }
  }
  return out;
}

template<typename T1, typename T2>
ImageD MatMul(const Image<T1>& A, const Image<T2>& B) {
  return MatMul<double, T1, T2>(A, B);
}

template<typename T1, typename T2>
ImageI MatMulI(const Image<T1>& A, const Image<T2>& B) {
  return MatMul<int, T1, T2>(A, B);
}

template<typename T>
inline Image<T> Identity(const size_t N) {
  Image<T> out(N, N, T());
  for (int i = 0; i < N; ++i) {
    out.Row(i)[i] = static_cast<T>(1.0);
  }
  return out;
}

inline ImageD Diagonal(const ImageD& d) {
  PIK_ASSERT(d.ysize() == 1);
  ImageD out(d.xsize(), d.xsize(), 0.0);
  for (size_t k = 0; k < d.xsize(); ++k) {
    out.Row(k)[k] = d.Row(0)[k];
  }
  return out;
}

// Computes c, s such that c^2 + s^2 = 1 and
//   [c -s] [a0  b] [c  s]
//   [s  c] [b  a1] [-s c]
// is diagonal.
void Diagonalize2x2(const double a0, const double a1, const double b,
                    double* c, double* s);

// Computes c, s such that c^2 + s^2 = 1 and
//   [c -s] [x] = [ * ]
//   [s  c] [y]   [ 0 ]
void GivensRotation(const double x, const double y, double* c, double* s);

// U = U * Givens(i, j, c, s)
void RotateMatrixCols(ImageD* const PIK_RESTRICT U,
                      int i, int j, double c, double s);

// U = Transpose(Givens(i, j, c, s)) * U
void RotateMatrixRows(ImageD* const PIK_RESTRICT U,
                      int i, int j, double c, double s);

// A is symmetric, U is orthogonal, T is tri-diagonal and
// A = U * T * Transpose(U).
void ConvertToTridiagonal(const ImageD& A,
                          ImageD* const PIK_RESTRICT T,
                          ImageD* const PIK_RESTRICT U);

// A is symmetric, U is orthogonal, and A = U * Diagonal(diag) * Transpose(U).
void ConvertToDiagonal(const ImageD& A,
                       ImageD* const PIK_RESTRICT diag,
                       ImageD* const PIK_RESTRICT U);

// A is square matrix, Q is orthogonal, R is upper triangular and A = Q * R;
void ComputeQRFactorization(const ImageD& A,
                            ImageD* const PIK_RESTRICT Q,
                            ImageD* const PIK_RESTRICT R);

// Computes the Lenstra-Lenstra-Lovász lattice basis reduction of matrix A.
//   Lenstra, A. K.; Lenstra, H. W., Jr.; Lovász, L. "Factoring polynomials with
//   rational coefficients". Mathematische Annalen. 261 (4): 515–534
//
// A is square matrix, Q is orthogonal, R is reduced upper triangular,
// Z * ZI = I and Transpose(Q) * A * Z = R.
//
// The running time of the algorithm is O(N^4), where N=A.xsize().
void ComputeLLLReduction(const ImageD& A,
                         ImageD* const PIK_RESTRICT Q,
                         ImageD* const PIK_RESTRICT R,
                         ImageI* const PIK_RESTRICT Z,
                         ImageI* const PIK_RESTRICT ZI);

class LatticeOptimizer {
 public:
  void InitFromLatticeBasis(const ImageD& A);

  void InitFromQuadraticForm(const ImageD& A);

  void Search(const ImageD& y, ImageI* const PIK_RESTRICT z0) const;

 private:
  ImageD R_;
  ImageD Qt_;
  ImageI Z_;
  std::vector<double> Rd_;
  std::vector<double> iRd_;
};

// Given a square matrix A and target vector y, finds integer vector z0 that
// minimizes D(z) = ||y - A*z||^2 over all integer vectors.
void FindClosestLatticeVector(const ImageD& A, const ImageD& y,
                              ImageI* const PIK_RESTRICT z0);

// Given a positive definite symmetric matrix A and a target vector y, finds
// integer vector z0 that minimizes D(z) = Transpose(z-y)*A*(z-y) over all
// integer vectors.
void OptimizeIntegerQuadraticForm(const ImageD& A, const ImageD& y,
                                  Image<int>* const PIK_RESTRICT z0);

}  // namespace pik

#endif  // LINALG_H_
