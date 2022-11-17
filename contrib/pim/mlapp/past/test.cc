#include <stdio.h>
#include <omp.h>

int main() {
	printf("omp_num_threads: %d\n", omp_get_num_threads());
	printf("omp_max_threads: %d\n", omp_get_max_threads());
	return 0;
}
