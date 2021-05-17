#pragma once

#include "deglib.h"

namespace deglib {


  static inline float _mm256_reduce_add_ps(__m256 x) {
      /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
      const __m128 x128 =
          _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
      /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
      const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
      /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
      const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
      /* Conversion to float is a no-op on x86-64 */
      return _mm_cvtss_f32(x32);
  }

  static inline float L2SqrSIMD16ExtAlignedOpt(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    #ifndef _WINDOWS
      float *a = (const float *) __builtin_assume_aligned((float *) pVect1v, 32);
      float *b = (const float *) __builtin_assume_aligned((float *) pVect2v, 32);
    #else
      float *a = (float *) pVect1v;
      float *b = (float *) pVect2v;
    #endif

    size_t size = *((size_t *) qty_ptr);
    __m256 a_vec, b_vec, tmp_vec;
    __m256 sum = _mm256_setzero_ps();
    for (size_t j = 0; j < size; j+=16) {

      a_vec = _mm256_load_ps(a + j);
      b_vec = _mm256_load_ps(b + j);
      tmp_vec = _mm256_sub_ps(a_vec, b_vec);      
      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);

      a_vec = _mm256_load_ps(a + j + 8);            // load a_vec     
      b_vec = _mm256_load_ps(b + j + 8);            // load b_vec     
      tmp_vec = _mm256_sub_ps(a_vec, b_vec);        // a_vec - b_vec
      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum); // sum = (tmp_vec**2) + sum
    }

    // horizontal add sum
    return deglib::_mm256_reduce_add_ps(sum);
  }

static inline __m128 masked_read (size_t d, const float *x)
{
    assert (0 <= d && d < 4);
    PORTABLE_ALIGN16 float buf[4] = {0, 0, 0, 0};
    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


  static inline float L2SqrSIMD16ExtAlignedFaiss(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    #ifndef _WINDOWS
      float *a = (const float *) __builtin_assume_aligned((float *) pVect1v, 32);
      float *b = (const float *) __builtin_assume_aligned((float *) pVect2v, 32);
    #else
      float *x = (float *) pVect1v;
      float *y = (float *) pVect2v;
    #endif
    size_t d = *((size_t *) qty_ptr);

    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 16) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_fmadd_ps(a_m_b1, a_m_b1, msum1);
        d -= 8;

        __m256 mx2 = _mm256_loadu_ps (x); x += 8;
        __m256 my2 = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b2 = _mm256_sub_ps(mx2, my2);
        msum1 = _mm256_fmadd_ps(a_m_b2, a_m_b2, msum1);
        d -= 8;
    }

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        msum1 = _mm256_fmadd_ps(a_m_b1, a_m_b1, msum1);
        d -= 8;
    }

    __m128 msum2 = _mm_add_ps(_mm256_extractf128_ps(msum1, 1), _mm256_extractf128_ps(msum1, 0));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_fmadd_ps(a_m_b1, a_m_b1, msum2);
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum2 = _mm_fmadd_ps(a_m_b1, a_m_b1, msum2);
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
  }

  static inline float L2SqrSIMD16ExtAlignedNGT(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    #ifndef _WINDOWS
      float *a = (const float *) __builtin_assume_aligned((float *) pVect1v, 32);
      float *b = (const float *) __builtin_assume_aligned((float *) pVect2v, 32);
    #else
      float *a = (float *) pVect1v;
      float *b = (float *) pVect2v;
    #endif
    size_t size = *((size_t *) qty_ptr);

    const float *last = a + size;
#if defined(USE_AVX512)
      __m512 sum512 = _mm512_setzero_ps();
      while (a < last) {
	__m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
	sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v, v));
	a += 16;
	b += 16;
      }

      __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
      __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif defined(USE_AVX)
      __m256 sum256 = _mm256_setzero_ps();
      __m256 v;
      while (a < last) {
        v = _mm256_sub_ps(_mm256_load_ps(a), _mm256_load_ps(b));
        sum256 = _mm256_fmadd_ps(v, v, sum256);
        a += 8;
        b += 8;        
        v = _mm256_sub_ps(_mm256_load_ps(a), _mm256_load_ps(b));
        sum256 = _mm256_fmadd_ps(v, v, sum256);
        a += 8;
        b += 8;
      }
      __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif defined(USE_SSE)
      __m128 sum128 = _mm_setzero_ps();
      __m128 v;
      while (a < last) {
	v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
	sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
        a += 4;
        b += 4;
	v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
	sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
        a += 4;
        b += 4;
	v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
	sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
        a += 4;
        b += 4;
	v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
	sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
        a += 4;
        b += 4;
      }
#endif

    // sum128 = _mm_hadd_ps (sum128, sum128);
    // sum128 = _mm_hadd_ps (sum128, sum128);
    // return  _mm_cvtss_f32 (sum128);

    float PORTABLE_ALIGN32 f[4];
    _mm_store_ps(f, sum128);

    return f[0] + f[1] + f[2] + f[3];
  }

  static inline float L2SqrSIMD16ExtAlignedM256(const __m256 *pVect1v, const void *pVect2v, const void *qty_ptr) {
    #ifndef _WINDOWS
      float *b = (const float *) __builtin_assume_aligned((float *) pVect2v, 32);
    #else
      float *b = (float *) pVect2v;
    #endif
    size_t size = *((size_t *) qty_ptr);

    size_t niters = size / 8;
    __m256 a_vec, b_vec, tmp_vec;
    __m256 sum = _mm256_setzero_ps();
    for (size_t j = 0; j < niters; j+=2) {

      a_vec = pVect1v[j];
      b_vec = _mm256_load_ps(b + 8 * j);
      tmp_vec = _mm256_sub_ps(a_vec, b_vec);      
      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);

      a_vec = pVect1v[j+1];                         // load a_vec     
      b_vec = _mm256_load_ps(b + 8 * j + 8);        // load b_vec     
      tmp_vec = _mm256_sub_ps(a_vec, b_vec);        // a_vec - b_vec
      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum); // sum = (tmp_vec**2) + sum
    }

    // horizontal add sum
    return deglib::_mm256_reduce_add_ps(sum);
  }

  static inline float L2SqrSIMD16ExtAligned(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
      float *pVect1 = (float *) pVect1v;
      float *pVect2 = (float *) pVect2v;
      size_t qty = *((size_t *) qty_ptr);
      float PORTABLE_ALIGN32 TmpRes[8];
      size_t qty16 = qty >> 4;

      const float *pEnd1 = pVect1 + (qty16 << 4);

      __m256 diff, v1, v2;
      __m256 sum = _mm256_set1_ps(0);

      while (pVect1 < pEnd1) {
          v1 = _mm256_load_ps(pVect1);
          pVect1 += 8;
          v2 = _mm256_load_ps(pVect2);
          pVect2 += 8;
          diff = _mm256_sub_ps(v1, v2);
          sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

          v1 = _mm256_load_ps(pVect1);
          pVect1 += 8;
          v2 = _mm256_load_ps(pVect2);
          pVect2 += 8;
          diff = _mm256_sub_ps(v1, v2);
          sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
      }

      _mm256_store_ps(TmpRes, sum);
      return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
  }


    static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

    #if defined(USE_AVX)

        // Favor using AVX if available.
        static float L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            float *pVect1 = (float *) pVect1v;
            float *pVect2 = (float *) pVect2v;
            size_t qty = *((size_t *) qty_ptr);
            float PORTABLE_ALIGN32 TmpRes[8];
            size_t qty16 = qty >> 4;

            const float *pEnd1 = pVect1 + (qty16 << 4);

            __m256 diff, v1, v2;
            __m256 sum = _mm256_set1_ps(0);

            while (pVect1 < pEnd1) {
                v1 = _mm256_loadu_ps(pVect1);
                pVect1 += 8;
                v2 = _mm256_loadu_ps(pVect2);
                pVect2 += 8;
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

                v1 = _mm256_loadu_ps(pVect1);
                pVect1 += 8;
                v2 = _mm256_loadu_ps(pVect2);
                pVect2 += 8;
                diff = _mm256_sub_ps(v1, v2);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
            }

            _mm256_store_ps(TmpRes, sum);
            return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
        }

    #elif defined(USE_SSE)

        static float L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            float *pVect1 = (float *) pVect1v;
            float *pVect2 = (float *) pVect2v;
            size_t qty = *((size_t *) qty_ptr);
            float PORTABLE_ALIGN32 TmpRes[8];
            size_t qty16 = qty >> 4;

            const float *pEnd1 = pVect1 + (qty16 << 4);

            __m128 diff, v1, v2;
            __m128 sum = _mm_set1_ps(0);

            while (pVect1 < pEnd1) {
                //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
                v1 = _mm_loadu_ps(pVect1);
                pVect1 += 4;
                v2 = _mm_loadu_ps(pVect2);
                pVect2 += 4;
                diff = _mm_sub_ps(v1, v2);
                sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

                v1 = _mm_loadu_ps(pVect1);
                pVect1 += 4;
                v2 = _mm_loadu_ps(pVect2);
                pVect2 += 4;
                diff = _mm_sub_ps(v1, v2);
                sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

                v1 = _mm_loadu_ps(pVect1);
                pVect1 += 4;
                v2 = _mm_loadu_ps(pVect2);
                pVect2 += 4;
                diff = _mm_sub_ps(v1, v2);
                sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

                v1 = _mm_loadu_ps(pVect1);
                pVect1 += 4;
                v2 = _mm_loadu_ps(pVect2);
                pVect2 += 4;
                diff = _mm_sub_ps(v1, v2);
                sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
            }

            _mm_store_ps(TmpRes, sum);
            return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        }
    #endif

    #if defined(USE_SSE) || defined(USE_AVX)
        static float L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            size_t qty = *((size_t *) qty_ptr);
            size_t qty16 = qty >> 4 << 4;
            float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
            float *pVect1 = (float *) pVect1v + qty16;
            float *pVect2 = (float *) pVect2v + qty16;

            size_t qty_left = qty - qty16;
            float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
            return (res + res_tail);
        }
    #endif


    #ifdef USE_SSE
        static float L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            float PORTABLE_ALIGN32 TmpRes[8];
            float *pVect1 = (float *) pVect1v;
            float *pVect2 = (float *) pVect2v;
            size_t qty = *((size_t *) qty_ptr);


            size_t qty4 = qty >> 2;

            const float *pEnd1 = pVect1 + (qty4 << 2);

            __m128 diff, v1, v2;
            __m128 sum = _mm_set1_ps(0);

            while (pVect1 < pEnd1) {
                v1 = _mm_loadu_ps(pVect1);
                pVect1 += 4;
                v2 = _mm_loadu_ps(pVect2);
                pVect2 += 4;
                diff = _mm_sub_ps(v1, v2);
                sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
            }
            _mm_store_ps(TmpRes, sum);
            return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        }

        static float L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            size_t qty = *((size_t *) qty_ptr);
            size_t qty4 = qty >> 2 << 2;

            float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
            size_t qty_left = qty - qty4;

            float *pVect1 = (float *) pVect1v + qty4;
            float *pVect2 = (float *) pVect2v + qty4;
            float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

            return (res + res_tail);
        }
    #endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        L2Space(const size_t dim) {
            fstdistfunc_ = L2Sqr;
            #if defined(USE_SSE) || defined(USE_AVX)
                if (dim % 16 == 0)
                    fstdistfunc_ = L2SqrSIMD16Ext;
                else if (dim % 4 == 0)
                    fstdistfunc_ = L2SqrSIMD4Ext;
                else if (dim > 16)
                    fstdistfunc_ = L2SqrSIMD16ExtResiduals;
                else if (dim > 4)
                    fstdistfunc_ = L2SqrSIMD4ExtResiduals;
            #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        const size_t get_data_size() const {
            return data_size_;
        }

        const DISTFUNC<float> get_dist_func() const {
            return fstdistfunc_;
        }

        const void *get_dist_func_param() const {
            return &dim_;
        }

        ~L2Space() {}
    };

}  // namespace deglib