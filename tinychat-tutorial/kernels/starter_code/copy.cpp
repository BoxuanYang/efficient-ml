#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
// Compute C(float) = A(INT8 activation) B^T(INT4 weight)
void MatmulOperator::mat_mul_reference(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float acc = 0;
            // Compute each block, C[row][col] += A[row][ch] * B^T[col][ch]
            for (int ch = 0; ch < k;) {
                // pointer of the int4 weights, load B[col][ch]
                uint8_t *w_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                // pointer of the int8 activation, load A[row][ch]
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // scale of weight
                float s_w = params->scales[(col * k + ch) / block_size];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];

                // order of weights with QM_x86:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
                // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         256 bit
                // process 32 bytes of weigths (256 bit) = 2 blocks
                // intermediate variable to store sum of integer multiplication and accumulation

                float s_w_2nd = params->scales[(col * k + ch) / block_size + 1];
                float s_a_2nd = params->scales[(row * k + ch) / block_size + 1];

                int intermediate_sum = 0;
                int intermediate_sum_2nd = 0;
                // iterate through all pairs of weight
                for(int qj = 0; qj < 32; qj++){
                    uint8_t packed_weight = w_int4[qj];

                    signed char weight_1st = (packed_weight & 0xF) - 8.0;
                    signed char weight_2nd = (packed_weight >> 4) - 8.0;

                    intermediate_sum += weight_1st * a_int8[qj];
                    intermediate_sum_2nd += weight_2nd * a_int8[qj + 32];
                }

                acc += intermediate_sum * s_a * s_w;
                acc + intermediate_sum_2nd * s_a_2nd * s_w_2nd;

                ch += block_size * 2;



            }
            C->data_ptr[row * n + col] = acc;
        }
    }
};
}


