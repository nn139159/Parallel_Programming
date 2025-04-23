#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float maxValue = _pp_vset_float(9.999999f);
  __pp_vec_float x, result;
  __pp_vec_int y;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_mask maskValus, maskExponents, maskIsNegative, maskIsNotNegative;
  __pp_mask load_mask = _pp_init_ones();

  auto clampedExpVectorLoop = [&](int index) {
    _pp_vload_float(x, values + index, load_mask); // float x = values[i];
    _pp_vload_int(y, exponents + index, load_mask); // int y = exponents[i];

    _pp_veq_int(maskIsNegative, y, zero, load_mask);  // if (y == 0)
    _pp_vset_float(result, 1.f, maskIsNegative);  // output[i] = 1.f;

    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {
    _pp_vmove_float(result, x, maskIsNotNegative);  // float result = x;
    _pp_vsub_int(y, y, one, maskIsNotNegative); // y - 1
    _pp_vgt_int(maskExponents, y, zero, load_mask); // count = y
    while (_pp_cntbits(maskExponents) > 0) {  // while (count > 0)
      _pp_vmult_float(result, result, x, maskExponents);  // result *= x;
  
      _pp_vsub_int(y, y, one, maskExponents); // count--;
      _pp_vgt_int(maskExponents, y, zero, maskExponents);
    }
  
    _pp_vgt_float(maskValus, result, maxValue, load_mask); // if (result > 9.999999f)
    _pp_vmove_float(result, maxValue, maskValus); // result = 9.999999f;
  
    _pp_vstore_float(output + index, result, load_mask); // output[i] = result;
  };

  int i = 0;
  for (;i < N - VECTOR_WIDTH; i += VECTOR_WIDTH) 
  {
    clampedExpVectorLoop(i);
  }

  load_mask = _pp_init_ones(N - i);
  clampedExpVectorLoop(i);
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float sum;
  int vw = VECTOR_WIDTH;
  __pp_vec_float x;
  __pp_vec_float result = _pp_vset_float(0.f);
  __pp_mask resultMask = _pp_init_ones(1);
  __pp_mask maskAll = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll); // x = values[i]
    _pp_vadd_float(result, result, x, maskAll); // sum += x;
  }
  
  while(vw > 1) {
    _pp_hadd_float(result, result); // [0 1 2 3] -> [0+1 0+1 2+3 2+3]
    _pp_interleave_float(result, result); // [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
    vw >>= 1;
  }

  _pp_vstore_float(&sum, result, resultMask);
  return sum;
}

