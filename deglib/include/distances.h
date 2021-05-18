#pragma once

#include "deglib.h"

namespace deglib {



static inline float L2SqrSIMD16ExtAlignedNGT(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
{  
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
    __m512 v = _mm512_sub_ps(_mm512_load_ps(a), _mm512_load_ps(b));
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
    v = _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b));
    sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
          a += 4;
          b += 4;
    v = _mm_sub_ps(_mm_load_ps(a), _mm_loadu_ps(b));
    sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
          a += 4;
          b += 4;
    v = _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b));
    sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
          a += 4;
          b += 4;
    v = _mm_sub_ps(_mm_load_ps(a), _mm_load_ps(b));
    sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
          a += 4;
          b += 4;
  }
#endif

    float PORTABLE_ALIGN32 f[4];
    _mm_store_ps(f, sum128);

    return f[0] + f[1] + f[2] + f[3];
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