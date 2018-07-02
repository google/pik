#include "linalg.h"

#include <string.h>
#include <climits>
#include <queue>
#include <utility>
#include <vector>

#include "status.h"

namespace pik {

const double kPi = 3.1415926535897932;

void AssertSymmetric(const ImageD& A) {
#if defined(PIK_ENABLE_ASSERT)
  PIK_ASSERT(A.xsize() == A.ysize());
  for (size_t i = 0; i < A.xsize(); ++i) {
    for (size_t j = i + 1; j < A.xsize(); ++j) {
      PIK_ASSERT(std::abs(A.Row(i)[j] - A.Row(j)[i]) < 1e-15);
    }
  }
#endif
}

void Diagonalize2x2(const double a0, const double a1, const double b, double* c,
                    double* s) {
  if (std::abs(b) < 1e-15) {
    *c = 1.0;
    *s = 0.0;
    return;
  }
  double phi = std::atan2(2 * b, a1 - a0);
  double theta = b > 0.0 ? 0.5 * phi : 0.5 * phi + kPi;
  *c = std::cos(theta);
  *s = std::sin(theta);
}

void GivensRotation(const double x, const double y, double* c, double* s) {
  if (y == 0.0) {
    *c = x < 0.0 ? -1.0 : 1.0;
    *s = 0.0;
  } else {
    const double h = std::hypot(x, y);
    const double d = 1.0 / h;
    *c = x * d;
    *s = -y * d;
  }
}

void RotateMatrixCols(ImageD* const PIK_RESTRICT U, int i, int j, double c,
                      double s) {
  PIK_ASSERT(U->xsize() == U->ysize());
  const size_t N = U->xsize();
  double* const PIK_RESTRICT u_i = U->Row(i);
  double* const PIK_RESTRICT u_j = U->Row(j);
  std::vector<double> rot_i, rot_j;
  rot_i.reserve(N);
  rot_j.reserve(N);
  for (size_t k = 0; k < N; ++k) {
    rot_i.push_back(u_i[k] * c - u_j[k] * s);
    rot_j.push_back(u_i[k] * s + u_j[k] * c);
  }
  for (size_t k = 0; k < N; ++k) {
    u_i[k] = rot_i[k];
    u_j[k] = rot_j[k];
  }
}

void RotateMatrixRows(ImageD* const PIK_RESTRICT U, int i, int j, double c,
                      double s) {
  PIK_ASSERT(U->xsize() == U->ysize());
  const size_t N = U->xsize();
  for (size_t k = 0; k < N; ++k) {
    double* const PIK_RESTRICT row = U->Row(k);
    const double r_i = row[i] * c - row[j] * s;
    const double r_j = row[i] * s + row[j] * c;
    row[i] = r_i;
    row[j] = r_j;
  }
}

void HouseholderReflector(const size_t N, const double* x, double* u) {
  const double sigma = x[0] <= 0.0 ? 1.0 : -1.0;
  u[0] = x[0] - sigma * std::sqrt(DotProduct(N, x, x));
  for (size_t k = 1; k < N; ++k) {
    u[k] = x[k];
  }
  double u_norm = 1.0 / std::sqrt(DotProduct(N, u, u));
  for (size_t k = 0; k < N; ++k) {
    u[k] *= u_norm;
  }
}

void ConvertToTridiagonal(const ImageD& A, ImageD* const PIK_RESTRICT T,
                          ImageD* const PIK_RESTRICT U) {
  AssertSymmetric(A);
  const size_t N = A.xsize();
  *U = Identity<double>(A.xsize());
  *T = CopyImage(A);
  std::vector<ImageD> u_stack;
  for (size_t k = 0; k + 2 < N; ++k) {
    if (DotProduct(N - k - 2, &T->Row(k)[k + 2], &T->Row(k)[k + 2]) > 1e-15) {
      ImageD u(N, 1, 0.0);
      HouseholderReflector(N - k - 1, &T->Row(k)[k + 1], &u.Row(0)[k + 1]);
      ImageD v = MatMul(*T, u);
      double scale = DotProduct(u, v);
      v = LinComb(2.0, v, -2.0 * scale, u);
      SubtractFrom(MatMul(u, Transpose(v)), T);
      SubtractFrom(MatMul(v, Transpose(u)), T);
      u_stack.emplace_back(std::move(u));
    }
  }
  while (!u_stack.empty()) {
    const ImageD& u = u_stack.back();
    ImageD v = MatMul(Transpose(*U), u);
    SubtractFrom(ScaleImage(2.0, MatMul(u, Transpose(v))), U);
    u_stack.pop_back();
  }
}

double WilkinsonShift(const double a0, const double a1, const double b) {
  const double d = 0.5 * (a0 - a1);
  if (d == 0.0) {
    return a1 - std::abs(b);
  }
  const double sign_d = d > 0.0 ? 1.0 : -1.0;
  return a1 - b * b / (d + sign_d * std::hypot(d, b));
}

void ImplicitQRStep(ImageD* const PIK_RESTRICT U, double* const PIK_RESTRICT a,
                    double* const PIK_RESTRICT b, int m0, int m1) {
  PIK_ASSERT(m1 - m0 > 2);
  double x = a[m0] - WilkinsonShift(a[m1 - 2], a[m1 - 1], b[m1 - 1]);
  double y = b[m0 + 1];
  for (int k = m0; k < m1 - 1; ++k) {
    double c, s;
    GivensRotation(x, y, &c, &s);
    const double w = c * x - s * y;
    const double d = a[k] - a[k + 1];
    const double z = (2 * c * b[k + 1] + d * s) * s;
    a[k] -= z;
    a[k + 1] += z;
    b[k + 1] = d * c * s + (c * c - s * s) * b[k + 1];
    x = b[k + 1];
    if (k > m0) {
      b[k] = w;
    }
    if (k < m1 - 2) {
      y = -s * b[k + 2];
      b[k + 2] *= c;
    }
    RotateMatrixCols(U, k, k + 1, c, s);
  }
}

void ScanInterval(const double* const PIK_RESTRICT a,
                  const double* const PIK_RESTRICT b, int istart,
                  const int iend, const double eps,
                  std::deque<std::pair<int, int> >* intervals) {
  for (int k = istart; k < iend; ++k) {
    if ((k + 1 == iend) ||
        std::abs(b[k + 1]) < eps * (std::abs(a[k]) + std::abs(a[k + 1]))) {
      if (k > istart) {
        intervals->push_back(std::make_pair(istart, k + 1));
      }
      istart = k + 1;
    }
  }
}

void ConvertToDiagonal(const ImageD& A, ImageD* const PIK_RESTRICT diag,
                       ImageD* const PIK_RESTRICT U) {
  AssertSymmetric(A);
  const size_t N = A.xsize();
  ImageD T;
  ConvertToTridiagonal(A, &T, U);
  // From now on, the algorithm keeps the transformed matrix tri-diagonal,
  // so we only need to keep track of the diagonal and the off-diagonal entries.
  std::vector<double> a(N);
  std::vector<double> b(N);
  for (size_t k = 0; k < N; ++k) {
    a[k] = T.Row(k)[k];
    if (k > 0) b[k] = T.Row(k)[k - 1];
  }
  // Run the symmetric tri-diagonal QR algorithm with implicit Wilkinson shift.
  const double kEpsilon = 1e-14;
  std::deque<std::pair<int, int> > intervals;
  ScanInterval(&a[0], &b[0], 0, N, kEpsilon, &intervals);
  while (!intervals.empty()) {
    const int istart = intervals[0].first;
    const int iend = intervals[0].second;
    intervals.pop_front();
    if (iend == istart + 2) {
      double& a0 = a[istart];
      double& a1 = a[istart + 1];
      double& b1 = b[istart + 1];
      double c, s;
      Diagonalize2x2(a0, a1, b1, &c, &s);
      const double d = a0 - a1;
      const double z = (2 * c * b1 + d * s) * s;
      a0 -= z;
      a1 += z;
      b1 = 0.0;
      RotateMatrixCols(U, istart, istart + 1, c, s);
    } else {
      ImplicitQRStep(U, &a[0], &b[0], istart, iend);
      ScanInterval(&a[0], &b[0], istart, iend, kEpsilon, &intervals);
    }
  }
  *diag = ImageD(N, 1);
  double* const PIK_RESTRICT diag_row = diag->Row(0);
  for (size_t k = 0; k < N; ++k) {
    diag_row[k] = a[k];
  }
}

void ComputeQRFactorization(const ImageD& A, ImageD* const PIK_RESTRICT Q,
                            ImageD* const PIK_RESTRICT R) {
  PIK_ASSERT(A.xsize() == A.ysize());
  const size_t N = A.xsize();
  *Q = Identity<double>(N);
  *R = CopyImage(A);
  std::vector<ImageD> u_stack;
  for (size_t k = 0; k + 1 < N; ++k) {
    if (DotProduct(N - k - 1, &R->Row(k)[k + 1], &R->Row(k)[k + 1]) > 1e-15) {
      ImageD u(N, 1, 0.0);
      HouseholderReflector(N - k, &R->Row(k)[k], &u.Row(0)[k]);
      ImageD v = MatMul(Transpose(u), *R);
      SubtractFrom(ScaleImage(2.0, MatMul(u, v)), R);
      u_stack.emplace_back(std::move(u));
    }
  }
  while (!u_stack.empty()) {
    const ImageD& u = u_stack.back();
    ImageD v = MatMul(Transpose(u), *Q);
    SubtractFrom(ScaleImage(2.0, MatMul(u, v)), Q);
    u_stack.pop_back();
  }
}

// Multiplies *A with IGT(i, j, mu) from the right, where
// IGT(i, j, mu) = I + mu*e_i*Transpose(e_j).
template <typename T>
void ApplyIGTCol(const size_t i, const size_t j, int mu,
                 Image<T>* const PIK_RESTRICT A) {
  const T* const PIK_RESTRICT a_i = A->ConstRow(i);
  T* const PIK_RESTRICT a_j = A->Row(j);
  for (size_t k = 0; k < A->xsize(); ++k) {
    a_j[k] += mu * a_i[k];
  }
}

// Multiplies *A with IGT(i, j, mu) from the left.
template <typename T>
void ApplyIGTRow(const size_t i, const size_t j, int mu,
                 Image<T>* const PIK_RESTRICT A) {
  for (size_t k = 0; k < A->ysize(); ++k) {
    T* const PIK_RESTRICT a_k = A->Row(k);
    a_k[i] += mu * a_k[j];
  }
}

template <typename T>
void SwapMatrixCols(const size_t i, const size_t j,
                    Image<T>* const PIK_RESTRICT A) {
  T* const PIK_RESTRICT a_i = A->Row(i);
  T* const PIK_RESTRICT a_j = A->Row(j);
  for (size_t k = 0; k < A->xsize(); ++k) {
    std::swap(a_i[k], a_j[k]);
  }
}

template <typename T>
void SwapMatrixRows(const size_t i, const size_t j,
                    Image<T>* const PIK_RESTRICT A) {
  for (size_t k = 0; k < A->ysize(); ++k) {
    T* const PIK_RESTRICT a_k = A->Row(k);
    std::swap(a_k[i], a_k[j]);
  }
}

template <typename T>
void MirrorMatrixCol(const size_t i, Image<T>* const PIK_RESTRICT A) {
  T* const PIK_RESTRICT a_i = A->Row(i);
  for (size_t k = 0; k < A->xsize(); ++k) {
    a_i[k] = -a_i[k];
  }
}

template <typename T>
void MirrorMatrixRow(const size_t i, Image<T>* const PIK_RESTRICT A) {
  for (size_t k = 0; k < A->ysize(); ++k) {
    T* const PIK_RESTRICT a_k = A->Row(k);
    a_k[i] = -a_k[i];
  }
}

void ComputeLLLReduction(const ImageD& A, ImageD* const PIK_RESTRICT Q,
                         ImageD* const PIK_RESTRICT R,
                         ImageI* const PIK_RESTRICT Z,
                         ImageI* const PIK_RESTRICT ZI) {
  PIK_ASSERT(A.xsize() == A.ysize());
  const size_t N = A.xsize();
  ComputeQRFactorization(A, Q, R);
  *Z = Identity<int>(N);
  *ZI = Identity<int>(N);
  for (size_t i = 0; i < N; ++i) {
    if (R->Row(i)[i] < 0.0) {
      MirrorMatrixCol(i, R);
      MirrorMatrixCol(i, Z);
      MirrorMatrixRow(i, ZI);
    }
  }
  for (int k = 0; k + 1 < N; ++k) {
    double* const PIK_RESTRICT r_k = R->Row(k);
    double* const PIK_RESTRICT r_k1 = R->Row(k + 1);
    for (int i = k; i >= 0; --i) {
      int mu = static_cast<int>(std::round(r_k1[i] / R->Row(i)[i]));
      if (mu != 0) {
        ApplyIGTCol(i, k + 1, -mu, R);
        ApplyIGTCol(i, k + 1, -mu, Z);
        ApplyIGTRow(i, k + 1, mu, ZI);
      }
    }
    if (r_k[k] > std::hypot(r_k1[k], r_k1[k + 1])) {
      SwapMatrixCols(k, k + 1, R);
      SwapMatrixCols(k, k + 1, Z);
      SwapMatrixRows(k, k + 1, ZI);
      double c, s;
      GivensRotation(r_k[k], r_k[k + 1], &c, &s);
      RotateMatrixRows(R, k, k + 1, c, s);
      RotateMatrixCols(Q, k, k + 1, c, s);
      if (r_k1[k + 1] < 0.0) {
        MirrorMatrixCol(k + 1, R);
        MirrorMatrixCol(k + 1, Z);
        MirrorMatrixRow(k + 1, ZI);
      }
      k = k > 0 ? k - 2 : -1;
    }
  }
}

void SearchLattice(const ImageD& R, const ImageD& y,
                   const std::vector<double>& Rd,
                   const std::vector<double>& iRd,
                   ImageI* const PIK_RESTRICT z0) {
  PIK_ASSERT(R.xsize() == R.ysize());
  PIK_ASSERT(R.xsize() == y.xsize());
  PIK_ASSERT(y.ysize() == 1);
  const size_t N = R.xsize();
  *z0 = ImageI(N, 1, 0);
  std::vector<int> z(N);
  std::vector<int> step(N);
  std::vector<double> center(N);
  double search_radius_sq = std::numeric_limits<double>::max();
  int k = N - 1;
  for (;;) {
    double sum = 0.0;
    for (int i = k + 1; i < N; ++i) {
      sum += R.Row(i)[k] * z[i];
    }
    center[k] = (y.Row(0)[k] - sum) * iRd[k];
    z[k] = std::round(center[k]);
    step[k] = center[k] < z[k] ? -1 : 1;
    for (;;) {
      double sumsq = 0.0;
      for (int i = k; i < N; ++i) {
        double v = Rd[i] * (z[i] - center[i]);
        sumsq += v * v;
      }
      if (sumsq > search_radius_sq) {
        if (k == N - 1) {
          // There are no more integer points within the search ellipsoid, the
          // integer vector that we found last was the optimal solution.
          return;
        }
        // The current integer vector [ * * * * zk, z(k+1), ..., z(N-1) ]
        // can not be extended to an integer point within the search ellipsoid,
        // so we cut this branch and go back to the previous level of the tree.
        ++k;
      } else if (k > 0) {
        // Go down one level in the tree.
        --k;
        break;
      } else {
        // We found an integer point in the search ellipsoid, update the
        // output vector and set the new search radius.
        for (int j = 0; j < N; ++j) {
          z0->Row(0)[j] = z[j];
        }
        search_radius_sq = sumsq;
        ++k;
      }
      // Iterate on the current level of the tree using the Schnorr-Euchner
      // enumeration order.
      z[k] += step[k];
      step[k] = -step[k] + (step[k] < 0 ? 1 : -1);
    }
  }
}

void LatticeOptimizer::InitFromLatticeBasis(const ImageD& A) {
  PIK_ASSERT(A.xsize() == A.ysize());
  const size_t N = A.xsize();
  ImageI ZI;
  ComputeLLLReduction(A, &Qt_, &R_, &Z_, &ZI);
  Qt_ = Transpose(Qt_);
  Rd_.reserve(N);
  iRd_.reserve(N);
  for (int i = 0; i < N; ++i) {
    const double r_ii = R_.Row(i)[i];
    Rd_.push_back(r_ii);
    iRd_.push_back(1.0 / r_ii);
  }
}

void LatticeOptimizer::InitFromQuadraticForm(const ImageD& A) {
  PIK_ASSERT(A.xsize() == A.ysize());
  const size_t N = A.xsize();
  ImageD Q, d;
  ConvertToDiagonal(A, &d, &Q);
  for (size_t k = 0; k < N; ++k) {
    PIK_ASSERT(d.Row(0)[k] > 0.0);
    d.Row(0)[k] = std::sqrt(d.Row(0)[k]);
  }
  ImageD B = MatMul(Diagonal(d), Transpose(Q));
  InitFromLatticeBasis(B);
  Qt_ = MatMul(Qt_, B);
}

void LatticeOptimizer::Search(const ImageD& y,
                              ImageI* const PIK_RESTRICT z0) const {
  PIK_ASSERT(R_.xsize() == y.xsize());
  PIK_ASSERT(y.ysize() == 1);
  ImageD yt = MatMul(Qt_, y);
  SearchLattice(R_, yt, Rd_, iRd_, z0);
  *z0 = MatMulI(Z_, *z0);
}

void FindClosestLatticeVector(const ImageD& A, const ImageD& y,
                              ImageI* const PIK_RESTRICT z0) {
  LatticeOptimizer lattice;
  lattice.InitFromLatticeBasis(A);
  lattice.Search(y, z0);
}

void OptimizeIntegerQuadraticForm(const ImageD& A, const ImageD& y,
                                  ImageI* const PIK_RESTRICT z0) {
  LatticeOptimizer lattice;
  lattice.InitFromQuadraticForm(A);
  lattice.Search(y, z0);
}

}  // namespace pik
