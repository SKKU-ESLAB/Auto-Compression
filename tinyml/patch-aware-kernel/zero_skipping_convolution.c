/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_s8_kernel3_inputch3_stride2_pad1.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"

#include "stm32f7xx_hal.h"

int start, end;
char buf[100];
/*
convolve_s8_kernel3_inputch3_stride2_pad1_fold(&buffer0[6400],40,40,
		3,(const q7_t*) weight0,bias0,
		shift0,multiplier0,
		-128,128,
		-128,127,&buffer0[0],
		20,20,
		16,sbuf,kbuf,-128, FoldWeight0);
*/
tinyengine_status convolve_s8_kernel3_inputch3_stride2_pad1_fold(const q7_t *input, const uint16_t input_x, const uint16_t input_y,
		const uint16_t input_ch, const q7_t *kernel, const int32_t *bias,
		const int32_t *output_shift, const int32_t *output_mult,
		const int32_t output_offset, const int32_t input_offset,
		const int32_t output_activation_min,
		const int32_t output_activation_max, q7_t *output,
		const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf, q15_t *kbuf, q7_t pad_value, const q7_t* fold_weight) {
	const int kernel_y = 3;
	const int kernel_x = 3;

	int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

	/* Generate two columns from the input tensor a GEMM computation */
	q15_t *two_column_buf = runtime_buf;
	q7_t *out = output;
	int stride = 1;
	q15_t pad16 = pad_value;
	const int16_t inoff16 = input_offset;
	q15_t pad_out = pad16 + inoff16;
	q31_t pad_out_q15x2 = __PKHBT(pad_out, pad_out, 16);
	q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);
	// zero skipping
	const q7_t *ip_a0 = kernel;
	const q7_t* ip_tb0 = fold_weight;
	const q7_t* ip_bb0= fold_weight + 288;

	// Memory 
	// [top border (Cout x 2 x 3 x 3 [18] (Cin)) | main kernel (Cout x 3 x 3 x 3) [27] | bot kernel (Cout x 2 x 3 x 3) [18] ]
	// --> 21 x 3 --> 63
	//DWT->CYCCNT = 0; start = DWT->CYCCNT;
	for (int i = 0; i < output_ch; i += 2) {
		q15_t *dst1 = &kbuf[i * 27]; //each q31_t store 2 elements
		q15_t *dst2 = dst1 + 27;

		const q7_t *ip_a1 = ip_a0 + 27;
		const q7_t* ip_tb1 = ip_tb0 + 18;
		const q7_t* ip_bb1 = ip_bb0 + 18;

		q31_t *dst1_31 = dst1;
		q31_t *dst2_31 = dst2;


		//18 for each output_ch for top kernel
		ip_tb0 = read_and_pad(ip_tb0, &dst1_31[0], &dst1_31[1]);
		ip_tb1 = read_and_pad(ip_tb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_tb0 = read_and_pad(ip_tb0, &dst1_31[0], &dst1_31[1]);
		ip_tb1 = read_and_pad(ip_tb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_tb0 = read_and_pad(ip_tb0, &dst1_31[0], &dst1_31[1]);
		ip_tb1 = read_and_pad(ip_tb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_tb0 = read_and_pad(ip_tb0, &dst1_31[0], &dst1_31[1]);
		ip_tb1 = read_and_pad(ip_tb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		dst1 = dst1_31;
		dst2 = dst2_31;
		dst1[0] = *ip_tb0++;
		dst1[1] = *ip_tb0++;
		dst2[0] = *ip_tb1++;
		dst2[1] = *ip_tb1++;

		dst1_31 = dst1 + 2;
		dst2_31 = dst2 + 2;

		//27 for each output_ch for mid kernel
		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_a0 = read_and_pad(ip_a0, &dst1_31[0], &dst1_31[1]);
		ip_a1 = read_and_pad(ip_a1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;
		//25, 26, 27
		dst1 = dst1_31;
		dst2 = dst2_31;
		dst1[0] = *ip_a0++;
		dst1[1] = *ip_a0++;
		dst1[2] = *ip_a0++;
		dst2[0] = *ip_a1++;
		dst2[1] = *ip_a1++;
		dst2[2] = *ip_a1++;

		dst1_31 = dst1 + 3;
		dst2_31 = dst2 + 3;

		//18 for each output_ch for top kernel
		ip_bb0 = read_and_pad(ip_bb0, &dst1_31[0], &dst1_31[1]);
		ip_bb1  = read_and_pad(ip_bb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_bb0 = read_and_pad(ip_bb0, &dst1_31[0], &dst1_31[1]);
		ip_bb1 = read_and_pad(ip_bb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_bb0 = read_and_pad(ip_bb0, &dst1_31[0], &dst1_31[1]);
		ip_bb1 = read_and_pad(ip_bb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		ip_bb0 = read_and_pad(ip_bb0, &dst1_31[0], &dst1_31[1]);
		ip_bb1 = read_and_pad(ip_bb1, &dst2_31[0], &dst2_31[1]);
		dst1_31 += 2;
		dst2_31 += 2;

		dst1 = dst1_31;
		dst2 = dst2_31;
		dst1[0] = *ip_tb0++;
		dst1[1] = *ip_tb0++;
		dst2[0] = *ip_tb1++;
		dst2[1] = *ip_tb1++;

		/* skip row */
		ip_a0 += 27;

		ip_tb0 += 18;
		ip_bb0 += 18;
	}
	//end = DWT->CYCCNT;sprintf(buf, "Kernel Copy %d\r\n", end - start);printLog(buf);    DWT->CYCCNT = 0;start = DWT->CYCCNT;
	int input_row_offset = 3 * input_x;
	q8_t * src = input;
	q8_t * src1 = src + input_row_offset;
	q8_t* src2;
	q8_t* src3;

	const q15_t *col_buffer = two_column_buf;
	q15_t* dst = col_buffer;
	q15_t* dst2 = dst + 6;
	q15_t* dst3;


	q31_t in_q7x4;
	q31_t in_q15x2_1;
	q31_t in_q15x2_2;
	q31_t out_q15x2_1;
	q31_t out_q15x2_2;

	// Top Left
	// 2 * 3 + 2 * 3 --> 12
	// 6 = 4 + 2;

	q7_q15_offset_ele(src, dst);
	*dst++ = *src++ + input_offset;
	*dst++ = *src++ + input_offset;

	q7_q15_offset_ele(src2, dst2);
	*dst2++ = *src2++ + input_offset;
	*dst2++ = *src2++ + input_offset;


	const q7_t *ker_a = fold_weight + 576;
	int i;
	for(i = 0; i < output_ch; i++) {
		q31_t sum = bias[i];
		const q15_t *ip_as_col = runtime_buf;
		uint16_t col_count = input_ch;
		while(col_count) {
			q31_t ker_a1, ker_a2;
			q31_t ip_b1, ip_b2;

			ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

			ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
			sum = __SMLAD(ker_a1, ip_b1, sum);
			ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
			sum = __SMLAD(ker_a2, ip_b2, sum);

			col_count--;
		}

		col_count = input_ch * kernel_y * kernel_x & 0x3;
		while (col_count) {
			q7_t ker_a1 = *ker_a++;
			q15_t ip_b1 = *ip_as_col++;
			sum += ker_a1 * ip_b1;
			col_count--;
		}

		sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
		sum += output_offset;
		sum = MAX(sum, output_activation_min);
		sum = MIN(sum, output_activation_max);
		*out++ = (q7_t) sum;
	}

	two_column_buf = runtime_buf;

	//end = DWT->CYCCNT;sprintf(buf, "TOP TOP %d\r\n", end - start);printLog(buf);    DWT->CYCCNT = 0;start = DWT->CYCCNT;
	for(i_out_x = 1; i_out_x < output_x; i_out_x++) {
		const int16_t base_idx_x = (i_out_x * 2 - 1) * input_ch;
		src = input + base_idx_x;
		src2 = src + input_row_offset;
		q15_t* dst = two_column_buf;
		q15_t* dst2 = dst + 6;
		// Top read
		// 3 * 3 + 3 * 3 --> 18
		//4 * 2 + 1 = 9
		q7_q15_offset_ele(src, dst)
		q7_q15_offset_ele(src, dst)
		*dst++ = *src++ + input_offset;
		//4 * 2 + 1 = 9
		q7_q15_offset_ele(src2, dst2)
		q7_q15_offset_ele(src2, dst2)
		*dst2++ = *src2++ + input_offset;

		two_column_buf += 18;
		if (two_column_buf == runtime_buf + 36) {
			out = mat_mult_unloop18_s8_s16(kernel,
					runtime_buf, output_ch, output_shift, output_mult,
					output_offset, output_activation_min, output_activation_max,
					input_ch * 6, bias, out, kbuf);

			two_column_buf = runtime_buf;
		}
	}
	if((output_y & 1) == (input_y & 1)) {
		const q7_t *ker_a = kernel;
		int i;

		for (i = 0; i < output_ch; i++) {
			/* Load the accumulator with bias first */
			q31_t sum = bias[i];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const q15_t *ip_as_col = runtime_buf;

			/* 4 multiply and accumulates are done in one loop. */
			// thinking point
			uint16_t col_count = (input_ch * 3) >> 1;

			while (col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			/* Handle left over mac */
			col_count = (input_ch * 2 * 3) & 0x1;
			while (col_count) {
				q7_t ker_a1 = *ker_a++;
				q15_t ip_b1 = *ip_as_col++;
				sum += ker_a1 * ip_b1;
				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}
	}

	two_column_buf = runtime_buf;
	kbuf += 3 * 6 * 12;
	//


	//end = DWT->CYCCNT;sprintf(buf, "CONV3x3_TOP %d\r\n", end - start);printLog(buf);    DWT->CYCCNT = 0;start = DWT->CYCCNT;
	// Top compute
	//DWT->CYCCNT = 0; start = DWT->CYCCNT;
	for(i_out_y = 1; i_out_y < output_y; i_out_y++) {
			const int16_t base_idx_y = (i_out_y * 2 - 1) * input_row_offset;
			src = input + base_idx_y;
			src2 = src + input_row_offset;
			src3 = src2 + input_row_offset;
			dst = two_column_buf;
			dst2 = dst + 6;
			dst3 = dst2 + 6;
			// Top read
			// 3 * 3 + 3 * 3 --> 18
			//4 * 2 + 1 = 9
			q7_q15_offset_ele(src, dst)
			*dst++ = *src++ + input_offset;
			*dst++ = *src++ + input_offset;

			//4 * 2 + 1 = 9
			q7_q15_offset_ele(src2, dst2)
			*dst2++ = *src2++ + input_offset;
			*dst2++ = *src2++ + input_offset;

			q7_q15_offset_ele(src3, dst3)
			*dst3++ = *src3++ + input_offset;
			*dst3++ = *src3++ + input_offset;

			two_column_buf += 18;
			if (two_column_buf == runtime_buf + 36) {
				out = mat_mult_unloop18_s8_s16_stride(kernel,
						runtime_buf, output_ch, output_shift, output_mult,
						output_offset, output_activation_min, output_activation_max,
						input_ch * 6, bias, output_x, out, kbuf);

				two_column_buf = runtime_buf;
			}
		}
		if((output_y & 1) == (input_y & 1)) {
			const q7_t *ker_a = kernel;
			int i;

			for (i = 0; i < output_ch; i++) {
				/* Load the accumulator with bias first */
				q31_t sum = bias[i];

				/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
				const q15_t *ip_as_col = runtime_buf;

				/* 4 multiply and accumulates are done in one loop. */
				// thinking point
				uint16_t col_count = (input_ch * 3) >> 1;

				while (col_count) {
					q31_t ker_a1, ker_a2;
					q31_t ip_b1, ip_b2;

					ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

					ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
					sum = __SMLAD(ker_a1, ip_b1, sum);
					ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
					sum = __SMLAD(ker_a2, ip_b2, sum);

					col_count--;
				}
				/* Handle left over mac */
				col_count = (input_ch * 2 * 3) & 0x1;
				while (col_count) {
					q7_t ker_a1 = *ker_a++;
					q15_t ip_b1 = *ip_as_col++;
					sum += ker_a1 * ip_b1;
					col_count--;
				}

				sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
				sum += output_offset;
				sum = MAX(sum, output_activation_min);
				sum = MIN(sum, output_activation_max);
				*out++ = (q7_t) sum;
			}
		}
		two_column_buf = runtime_buf;
		out -= (output_y * output_x - output_x + 1) * output_ch;
	//end = DWT->CYCCNT;sprintf(buf, "CONV3x3_LEFT %d\r\n", end - start);printLog(buf);    DWT->CYCCNT = 0;start = DWT->CYCCNT;
	/*
	// topright
	if(input_x & 1) {
		const q15_t *col_buffer = two_column_buf;
		src = input + (input_x - 1) * input_ch;
		dst = col_buffer;
		dst2 = dst + 6;

		// Top Left
		// 2 * 3 + 2 * 3 --> 12
		q7_q15_offset_ele(src, dst);
		*dst++ = *src++ + input_offset;
		*dst++ = *src++ + input_offset;

		q7_q15_offset_ele(src2, dst2);
		*dst2++ = *src2++ + input_offset;
		*dst2++ = *src2++ + input_offset;

		const q7_t *ker_a = fold_weight + 768;
		for(i = 0; i < output_ch; i++) {
			q31_t sum = bias[i];
			const q15_t *ip_as_col = runtime_buf;
			uint16_t col_count = input_ch;
			while(col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			col_count = input_ch * kernel_y * kernel_x & 0x3;
			while (col_count) {
				q7_t ker_a1 = *ker_a++;
				q15_t ip_b1 = *ip_as_col++;
				sum += ker_a1 * ip_b1;
				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}
	}
	*/
	kbuf += 3 * 6 * 12;
	for(i_out_y = 1; i_out_y < output_y; i_out_y++) {
		const int16_t base_idx_y = (i_out_y * 2) - 1;

/*
		const q15_t *col_buffer = two_column_buf;
		src = input + (base_idx_y * input_x) * input_ch;
		src2 = src1 + input_row_offset;
		src3 = src2 + input_row_offset;
		dst = col_buffer;
		dst2 = dst + 6;
		dst3 = dst2 + 6;

		// Top Left
		// 2 * 3 + 2  * 3 + 2 * 3-->  18
		q7_q15_offset_ele(src, dst)
		*dst++ = *src++ + input_offset;
		*dst++ = *src++ + input_offset;

		q7_q15_offset_ele(src2, dst2)
		*dst2++ = *src2++ + input_offset;
		*dst2++ = *src2++ + input_offset;

		q7_q15_offset_ele(src3, dst3)
		*dst3++ = *src3++ + input_offset;
		*dst3++ = *src3++ + input_offset;

		const q7_t *ker_a = fold_weight + 960;
		for(i = 0; i < output_ch; i++) {
			q31_t sum = bias[i];
			const q15_t *ip_as_col = runtime_buf;
			uint16_t col_count = input_ch * 3 >> 1;
			while(col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			col_count = input_ch * kernel_y * kernel_x & 0x3;
			while (col_count) {
				q7_t ker_a1 = *ker_a++;
				q15_t ip_b1 = *ip_as_col++;
				sum += ker_a1 * ip_b1;
				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}
*/
		for(i_out_x = 1; i_out_x < output_x; i_out_x++) {

			const int16_t base_idx_x = (i_out_x * 2) - 1;
			const q15_t *col_buffer = two_column_buf;
			dst = col_buffer;
			dst2 = dst + 9;
			dst3 = dst2 + 9;
			src = input	+ (base_idx_y * input_x + base_idx_x) * input_ch;
			src2 = src + input_row_offset;
			src3 = src2 + input_row_offset;


			//4 * 2 = 8
			q7_q15_offset_ele(src, dst)
			q7_q15_offset_ele(src, dst)
			*dst++ = *src++ + input_offset;
			//
			q7_q15_offset_ele(src2, dst2)
			q7_q15_offset_ele(src2, dst2)
			*dst2++ = *src2++ + input_offset;
			//
			q7_q15_offset_ele(src3, dst3)
			q7_q15_offset_ele(src3, dst3)
			*dst3++ = *src3++ + input_offset;

			two_column_buf += 27;

			if (two_column_buf == runtime_buf + 2 * 27) {

				out = arm_nn_mat_mult_kernel3_input3_s8_s16_stride(kernel,
						runtime_buf, output_ch, output_shift, output_mult,
						output_offset, output_activation_min, output_activation_max,
						input_ch * kernel_y * kernel_x, bias, stride, out, kbuf);

				two_column_buf = runtime_buf;
				stride = 1;
			}
		}
		stride = 2;
	}

	if (two_column_buf != runtime_buf) {
		const q7_t *ker_a = kernel;
		int i;

		for (i = 0; i < output_ch; i++) {
			/* Load the accumulator with bias first */
			q31_t sum = bias[i];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const q15_t *ip_as_col = runtime_buf;

			/* 4 multiply and accumulates are done in one loop. */
			uint16_t col_count = (input_ch * kernel_y * kernel_x) >> 2;

			while (col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			/* Handle left over mac */
			col_count = input_ch * kernel_y * kernel_x & 0x3;
			while (col_count) {
				q7_t ker_a1 = *ker_a++;
				q15_t ip_b1 = *ip_as_col++;
				sum += ker_a1 * ip_b1;
				col_count--;
			}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}
	}
	//end = DWT->CYCCNT;sprintf(buf, "CONV3x3_MID %d\r\n", end - start);printLog(buf);    DWT->CYCCNT = 0;start = DWT->CYCCNT;
}
		/*
		if (output_x & 1) {
			const q15_t *col_buffer = two_column_buf;
			src = input + (base_idx_y * input_x +  input_x - 1) * input_ch;
			src2 = src1 + input_row_offset;
			src3 = src2 + input_row_offset;
			dst = col_buffer;
			dst2 = dst + 6;
			dst3 = dst2 + 6;

			// Top Left
			// 2 * 3 + 2 * 3 + 2 * 3-->  18
			q7_q15_offset_ele(src, dst)
			*dst++ = *src++ + input_offset;
			*dst++ = *src++ + input_offset;

			q7_q15_offset_ele(src2, dst2)
			*dst2++ = *src2++ + input_offset;
			*dst2++ = *src2++ + input_offset;

			q7_q15_offset_ele(src3, dst3)
			*dst3++ = *src3++ + input_offset;
			*dst3++ = *src3++ + input_offset;

			const q7_t *ker_a = fold_weight + 1248;
			for(i = 0; i < output_ch; i++) {
				q31_t sum = bias[i];
				const q15_t *ip_as_col = runtime_buf;
				uint16_t col_count = input_ch * 3 >> 1;
				while(col_count) {
					q31_t ker_a1, ker_a2;
					q31_t ip_b1, ip_b2;

					ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

					ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
					sum = __SMLAD(ker_a1, ip_b1, sum);
					ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
					sum = __SMLAD(ker_a2, ip_b2, sum);

					col_count--;
				}
				col_count = input_ch * kernel_y * kernel_x & 0x3;
				while (col_count) {
					q7_t ker_a1 = *ker_a++;
					q15_t ip_b1 = *ip_as_col++;
					sum += ker_a1 * ip_b1;
					col_count--;
				}

				sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
				sum += output_offset;
				sum = MAX(sum, output_activation_min);
				sum = MIN(sum, output_activation_max);
				*out++ = (q7_t) sum;
			}
		}
	}
*/

/*
	if(input_y & 1) {
		const int16_t base_idx_y = (input_y - 2) * input_x;
		const q15_t *col_buffer = two_column_buf;

		src = input	+ base_idx_y * input_x * input_ch;
		src2 = src + input_row_offset;

		q15_t* dst = col_buffer;
		q15_t* dst2 = dst + 6;

		q7_q15_offset_ele(src, dst)
		*dst++ = *src++ + input_offset;
		*dst++ = *src++ + input_offset;

		q7_q15_offset_ele(src2, dst2)
		*dst2++ = *src2++ + input_offset;
		*dst2++ = *src2++ + input_offset;

		const q7_t *ker_a = fold_weight + 1536;
		for(i = 0; i < output_ch; i++) {
			q31_t sum = bias[i];
			const q15_t *ip_as_col = runtime_buf;
			uint16_t col_count = input_ch;
			while(col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			//col_count = input_ch * kernel_y * kernel_x & 0x3;
			//while (col_count) {
			//	q7_t ker_a1 = *ker_a++;
			//	q15_t ip_b1 = *ip_as_col++;
			//	sum += ker_a1 * ip_b1;
			//	col_count--;
			//}

			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}

		for(i_out_x = 1; i_out_x < output_x - (output_x & 1); i_out_x++) {
			const int16_t base_idx_x = (i_out_x * 2) - 1;
			const q15_t *col_buffer = two_column_buf;
			src = input	+ (base_idx_y * input_x + base_idx_x) * input_ch;
			src2 = src + input_row_offset;

			dst = col_buffer;
			dst2 = dst + 9;
			//3 * 3 + 3 * 3 --> 18
			q7_q15_offset_ele(src, dst)
			q7_q15_offset_ele(src, dst)
			*dst++ = *src++ + input_offset;
			//
			q7_q15_offset_ele(src2, dst2)
			q7_q15_offset_ele(src2, dst2)
			*dst2++ = *src2++ + input_offset;
			//
			q7_q15_offset_ele(src3, dst3)
			q7_q15_offset_ele(src3, dst3)
			*dst3++ = *src3++ + input_offset;

			two_column_buf += 18;
			if (two_column_buf == runtime_buf + 2 * 18) {

				out = mat_mult_unloop18_s8_s16(kernel,
						runtime_buf, output_ch, output_shift, output_mult,
						output_offset, output_activation_min, output_activation_max,
						input_ch * 2 * 3, bias, out, kbuf + 16 * 45);
				two_column_buf = runtime_buf;
			}
		}


		/*
		col_buffer = two_column_buf;
		src = input + (input_x - 1) * input_ch;
		dst = col_buffer;
		dst2 = dst + 6;

		// Top Left
		// 2 * 3 + 2 * 3 --> 12
		q7_q15_offset_ele(src, dst);
		*dst++ = *src++ + input_offset;
		*dst++ = *src++ + input_offset;

		q7_q15_offset_ele(src2, dst2);
		*dst2++ = *src2++ + input_offset;
		*dst2++ = *src2++ + input_offset;

		ker_a = fold_weight + 1728;
		for(i = 0; i < output_ch; i++) {
			q31_t sum = bias[i];
			const q15_t *ip_as_col = runtime_buf;
			uint16_t col_count = input_ch;
			while(col_count) {
				q31_t ker_a1, ker_a2;
				q31_t ip_b1, ip_b2;

				ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

				ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, ip_b1, sum);
				ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, ip_b2, sum);

				col_count--;
			}
			col_count = input_ch;
			//while (col_count) {
			//	q7_t ker_a1 = *ker_a++;
			//	q15_t ip_b1 = *ip_as_col++;
			//	sum += ker_a1 * ip_b1;
			//	col_count--;
			//}
			sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
			sum += output_offset;
			sum = MAX(sum, output_activation_min);
			sum = MIN(sum, output_activation_max);
			*out++ = (q7_t) sum;
		}
	}
	*/

