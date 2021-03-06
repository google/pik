// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_LINALG_H_
#define PIK_LINALG_H_

// Linear algebra.

#include <stddef.h>
#include <cmath>
#include <vector>

#include "pik/image.h"
#include "pik/image_ops.h"
#include "pik/status.h"

namespace pik {

using ImageD = Image<double>;

inline double DotProduct(const size_t N, const double* const PIK_RESTRICT a,
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

template <typename Tout, typename Tin1, typename Tin2>
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

template <typename T1, typename T2>
ImageD MatMul(const Image<T1>& A, const Image<T2>& B) {
  return MatMul<double, T1, T2>(A, B);
}

template <typename T1, typename T2>
ImageI MatMulI(const Image<T1>& A, const Image<T2>& B) {
  return MatMul<int, T1, T2>(A, B);
}

// Computes A = B * C, with sizes rows*cols: A=ha*wa, B=wa*wb, C=ha*wb
template <typename T>
void MatMul(const T* a, const T* b, int ha, int wa, int wb, T* c) {
  std::vector<T> temp(wa);  // Make better use of cache lines
  for (int x = 0; x < wb; x++) {
    for (int z = 0; z < wa; z++) {
      temp[z] = b[z * wb + x];
    }
    for (int y = 0; y < ha; y++) {
      double e = 0;
      for (int z = 0; z < wa; z++) {
        e += a[y * wa + z] * temp[z];
      }
      c[y * wb + x] = e;
    }
  }
}

// Computes C = A + factor * B
template <typename T, typename F>
void MatAdd(const T* a, const T* b, F factor, int h, int w, T* c) {
  for (int i = 0; i < w * h; i++) {
    c[i] = a[i] + b[i] * factor;
  }
}

template <typename T>
inline Image<T> Identity(const size_t N) {
  Image<T> out(N, N);
  for (size_t i = 0; i < N; ++i) {
    T* PIK_RESTRICT row = out.Row(i);
    std::fill(row, row + N, 0);
    row[i] = static_cast<T>(1.0);
  }
  return out;
}

inline ImageD Diagonal(const ImageD& d) {
  PIK_ASSERT(d.ysize() == 1);
  ImageD out(d.xsize(), d.xsize());
  const double* PIK_RESTRICT row_diag = d.Row(0);
  for (size_t k = 0; k < d.xsize(); ++k) {
    double* PIK_RESTRICT row_out = out.Row(k);
    std::fill(row_out, row_out + d.xsize(), 0.0);
    row_out[k] = row_diag[k];
  }
  return out;
}

// Computes c, s such that c^2 + s^2 = 1 and
//   [c -s] [x] = [ * ]
//   [s  c] [y]   [ 0 ]
void GivensRotation(const double x, const double y, double* c, double* s);

// U = U * Givens(i, j, c, s)
void RotateMatrixCols(ImageD* const PIK_RESTRICT U, int i, int j, double c,
                      double s);

// A is symmetric, U is orthogonal, T is tri-diagonal and
// A = U * T * Transpose(U).
void ConvertToTridiagonal(const ImageD& A, ImageD* const PIK_RESTRICT T,
                          ImageD* const PIK_RESTRICT U);

// A is symmetric, U is orthogonal, and A = U * Diagonal(diag) * Transpose(U).
void ConvertToDiagonal(const ImageD& A, ImageD* const PIK_RESTRICT diag,
                       ImageD* const PIK_RESTRICT U);

// A is square matrix, Q is orthogonal, R is upper triangular and A = Q * R;
void ComputeQRFactorization(const ImageD& A, ImageD* const PIK_RESTRICT Q,
                            ImageD* const PIK_RESTRICT R);

// Inverts a 3x3 matrix in place
template <typename T>
void Inv3x3Matrix(T* matrix) {
  T temp[9];
  temp[0] = matrix[4] * matrix[8] - matrix[5] * matrix[7];
  temp[1] = matrix[2] * matrix[7] - matrix[1] * matrix[8];
  temp[2] = matrix[1] * matrix[5] - matrix[2] * matrix[4];
  temp[3] = matrix[5] * matrix[6] - matrix[3] * matrix[8];
  temp[4] = matrix[0] * matrix[8] - matrix[2] * matrix[6];
  temp[5] = matrix[2] * matrix[3] - matrix[0] * matrix[5];
  temp[6] = matrix[3] * matrix[7] - matrix[4] * matrix[6];
  temp[7] = matrix[1] * matrix[6] - matrix[0] * matrix[7];
  temp[8] = matrix[0] * matrix[4] - matrix[1] * matrix[3];
  T idet =
      1.0 / (matrix[0] * temp[0] + matrix[1] * temp[3] + matrix[2] * temp[6]);
  for (int i = 0; i < 9; i++) {
    matrix[i] = temp[i] * idet;
  }
}

// Solves system of linear equations A * X = B using the conjugate gradient
// method. Matrix a must be a n*n, symmetric and positive definite.
// Vectors b and x must have n elements
template <typename T>
void ConjugateGradient(const T* a, int n, const T* b, T* x) {
  std::vector<T> r(n);
  MatMul(a, x, n, n, 1, r.data());
  MatAdd(b, r.data(), -1, n, 1, r.data());
  std::vector<T> p = r;
  T rr;
  MatMul(r.data(), r.data(), 1, n, 1, &rr);  // inner product

  if (rr == 0) return;  // The initial values were already optimal

  for (int i = 0; i < n; i++) {
    std::vector<T> ap(n);
    MatMul(a, p.data(), n, n, 1, ap.data());
    T alpha;
    MatMul(r.data(), ap.data(), 1, n, 1, &alpha);
    // Normally alpha couldn't be zero here but if numerical issues caused it,
    // return assuming the solution is close.
    if (alpha == 0) return;
    alpha = rr / alpha;
    MatAdd(x, p.data(), alpha, n, 1, x);
    MatAdd(r.data(), ap.data(), -alpha, n, 1, r.data());

    T rr2;
    MatMul(r.data(), r.data(), 1, n, 1, &rr2);  // inner product
    if (rr2 < 1e-20) break;

    T beta = rr2 / rr;
    MatAdd(r.data(), p.data(), beta, 1, n, p.data());
    rr = rr2;
  }
}

// Computes optimal coefficients r to approximate points p with linear
// combination of functions f. The matrix f has h rows and w columns, r has h
// values, p has w values. h is the amount of functions, w the amount of points.
// Uses the finite element method and minimizes mean square error.
template <typename T>
void FEM(const T* f, int h, int w, const T* p, T* r) {
  // Compute "Gramian" matrix G = F * F^T
  // Speed up multiplication by using non-zero intervals in sparse F.
  std::vector<int> start(h);
  std::vector<int> end(h);
  for (int y = 0; y < h; y++) {
    start[y] = end[y] = 0;
    for (int x = 0; x < w; x++) {
      if (f[y * w + x] != 0) {
        start[y] = x;
        break;
      }
    }
    for (int x = w - 1; x >= 0; x--) {
      if (f[y * w + x] != 0) {
        end[y] = x + 1;
        break;
      }
    }
  }

  std::vector<T> g(h * h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x <= y; x++) {
      T v = 0;
      // Intersection of the two sparse intervals.
      int s = std::max(start[x], start[y]);
      int e = std::min(end[x], end[y]);
      for (int z = s; z < e; z++) {
        v += f[x * w + z] * f[y * w + z];
      }
      // Symmetric, so two values output at once
      g[y * h + x] = v;
      g[x * h + y] = v;
    }
  }

  // B vector: sum of each column of F multiplied by corresponding p
  std::vector<T> b(h, 0);
  for (int y = 0; y < h; y++) {
    T v = 0;
    for (int x = 0; x < w; x++) {
      v += f[y * w + x] * p[x];
    }
    b[y] = v;
  }

  ConjugateGradient(g.data(), h, b.data(), r);
}

}  // namespace pik

#endif  // PIK_LINALG_H_
