#include <iostream>
#include <random>
#include <pthread.h>
#include <atomic>
#include <immintrin.h>

struct Xoshiro256ppSIMD {
    // 4 groups of 4 uint64_t for parallel operations
    alignas(32) uint64_t s[4][4];

    __m256d normalize_pd = _mm256_set1_pd(1.0 / (1ULL << 63));
    __m256 normalize_ps = _mm256_set1_ps(1.0 / (1ULL << 31));
    // Emulate _mm256_cvtepi64_pd
    static inline __m256d avx2_cvtepi64_pd(__m256i v) {
        alignas(32) int64_t temp[4];
        _mm256_store_si256((__m256i*)temp, v);
        return _mm256_set_pd((double)temp[3], (double)temp[2], (double)temp[1], (double)temp[0]);
    }

    // Rotate left using AVX2
    static inline __m256i rotl(__m256i x, int k) {
        return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k));
    }

    // Initialize the seed
    void seed(uint64_t seed_val) {
        for (int i = 0; i < 4; ++i) {
	    s[i][0] = seed_val;
	    s[i][1] = seed_val ^ 0x9E3779B97F4A7C15;
	    s[i][2] = seed_val + 0xD1B54A32D192ED03;
	    s[i][3] = seed_val ^ (seed_val >> 33);
        }
    }

    // Parallelization produces 4 uint64
    __m256i next() {
        // x[i] = s[0][i] + s[3][i];
        __m256i x = _mm256_add_epi64(_mm256_load_si256((__m256i*)s[0]), _mm256_load_si256((__m256i*)s[3]));
        // result[i] = ((x[i] << 23) | (x[i] >> 41)) + s[0][i];
        __m256i result = _mm256_add_epi64(rotl(x, 23), _mm256_load_si256((__m256i*)s[0]));
        // t[i] = s[1][i] << 17;
        __m256i t = _mm256_slli_epi64(_mm256_load_si256((__m256i*)s[1]), 17);

        // Status Update
        // s[2][i] ^= s[0][i];
        _mm256_store_si256((__m256i*)s[2], _mm256_xor_si256(_mm256_load_si256((__m256i*)s[2]), _mm256_load_si256((__m256i*)s[0])));
        // s[3][i] ^= s[1][i];
        _mm256_store_si256((__m256i*)s[3], _mm256_xor_si256(_mm256_load_si256((__m256i*)s[3]), _mm256_load_si256((__m256i*)s[1])));
        // s[1][i] ^= s[2][i];
        _mm256_store_si256((__m256i*)s[1], _mm256_xor_si256(_mm256_load_si256((__m256i*)s[1]), _mm256_load_si256((__m256i*)s[2])));
        // s[0][i] ^= s[3][i];
        _mm256_store_si256((__m256i*)s[0], _mm256_xor_si256(_mm256_load_si256((__m256i*)s[0]), _mm256_load_si256((__m256i*)s[3])));
        // s[2][i] ^= t[i];
        _mm256_store_si256((__m256i*)s[2], _mm256_xor_si256(_mm256_load_si256((__m256i*)s[2]), t));
        // s[3][i]  = (s[3][i] << 45) | (s[3][i] >> 19);
        _mm256_store_si256((__m256i*)s[3], rotl(_mm256_load_si256((__m256i*)s[3]), 45));

        return result;
    }
    
    // Convert to double output and normalize to (-1,1)
    __m256d next_double() {
        __m256d rand_pd = avx2_cvtepi64_pd(next());
        // rand_pd * (1.0 / (1ULL << 63))
	    return _mm256_mul_pd(rand_pd, normalize_pd);
    }
    
    // Convert to float output and normalize to (-1,1)
    __m256 next_float() {
	    __m256 rand_ps = _mm256_cvtepi32_ps(next());
        //  rand_ps * (1.0 / (1ULL << 31))
	    return _mm256_mul_ps(rand_ps, normalize_ps);
    }
};

struct ThreadData {
    	long long int tossesPerThread;
};


std::atomic<long long int> numberInCircle(0);
std::atomic<long long int> numActualTosses(0);
void* monte_carlo_pi(void* arg) {
	ThreadData* data = (ThreadData*)(arg);
	// Thread independent random number generator.
	std::random_device random;
	Xoshiro256ppSIMD gen;
	gen.seed(random());
	
	// Calculate the number of parallel loops, add 1 if there is a remainder, 
	// and make sure it is not less than the tosses input
	long long int loopSIMD = (data->tossesPerThread >> 3) + (data->tossesPerThread % 8 > 0);
	long long int localCount = 0;
	
	// AVX2 Vectorization
	__m256 ones = _mm256_set1_ps(1.0f);
	for (long long int i = 0; i < loopSIMD; ++i) {
		// Read into AVX register
		__m256 xVals = gen.next_float();
		__m256 yVals = gen.next_float();

		// Calculate x^2 + y^2
		__m256 xSqare = _mm256_mul_ps(xVals, xVals);
		__m256 ySqare = _mm256_mul_ps(yVals, yVals);
		__m256 distSqare = _mm256_add_ps(xSqare, ySqare);

		// mask less than or equal 1.0
		__m256 mask = _mm256_cmp_ps(distSqare, ones, _CMP_LE_OQ);

		//  Count the number of 1's in mask
		localCount += _mm_popcnt_u32(_mm256_movemask_ps(mask));
	}

	numActualTosses.fetch_add(loopSIMD << 3, std::memory_order_relaxed);
	numberInCircle.fetch_add(localCount, std::memory_order_relaxed);
	return nullptr;
}

double estimate_pi(long long int numTosses, int numThreads) {
	pthread_t threads[numThreads];
	ThreadData threadDatas[numThreads];
	long long int tossesPerThread = numTosses / numThreads;
	long long int remainder = numTosses % numThreads;

	for (int i = 0; i < numThreads; ++i) {
		threadDatas[i].tossesPerThread = tossesPerThread + (i < remainder ? 1 : 0);
		pthread_create(&threads[i], nullptr, monte_carlo_pi, &threadDatas[i]);
	}

	for (int i = 0; i < numThreads; ++i) {
		pthread_join(threads[i], nullptr);
	}

	return 4.0 * numberInCircle.load() / numActualTosses.load();
}

int main(int argc, char *argv[]) {
    int numThreads = strtol(argv[1], nullptr, 10);
    long long numTosses = strtoll(argv[2], nullptr, 10);
    double pi = estimate_pi(numTosses, numThreads);
    printf("%lf\n", pi);
    return 0;
}

