#include <pthread.h>
#include <iostream>

typedef struct {
	int num;
} thr_param_t;

pthread_t thr[4];
thr_param_t thr_param[4];
pthread_mutex_t mutex;
pthread_mutex_t print_mutex;
pthread_barrier_t thr_barrier;

static void* PrintNum(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int num = input->num;

	pthread_mutex_lock(&print_mutex);
	std::cout << "   " << num/10 << " " << num%10 << "\n";
	pthread_mutex_unlock(&print_mutex);

	pthread_barrier_wait(&thr_barrier);
	return (NULL);
}

void try2() {  // 4 times
	pthread_barrier_init(&thr_barrier, NULL, 5);
	for (int i=0; i<4; i++) {
		thr_param[i].num = i;
		pthread_create(&thr[i], NULL, PrintNum, (void*)&thr_param[i]);
	}
	pthread_barrier_wait(&thr_barrier);
	
	pthread_barrier_init(&thr_barrier, NULL, 5);
	for (int i=0; i<4; i++) {
		thr_param[i].num = i + 10;
		pthread_create(&thr[i], NULL, PrintNum, (void*)&thr_param[i]);
	}
	pthread_barrier_wait(&thr_barrier);
	
	pthread_barrier_init(&thr_barrier, NULL, 5);
	for (int i=0; i<4; i++) {
		thr_param[i].num = i + 20;
		pthread_create(&thr[i], NULL, PrintNum, (void*)&thr_param[i]);
	}
	pthread_barrier_wait(&thr_barrier);
	
	pthread_barrier_init(&thr_barrier, NULL, 5);
	for (int i=0; i<4; i++) {
		thr_param[i].num = i + 30;
		pthread_create(&thr[i], NULL, PrintNum, (void*)&thr_param[i]);
	}
	pthread_barrier_wait(&thr_barrier);
}	


void try1() {  // for + for
	for (int j=0; j<4; j++) {
		pthread_barrier_init(&thr_barrier, NULL, 4);
		for (int i=0; i<4; i++) {
			thr_param[i].num = j*10 + i;
			pthread_create(&thr[i], NULL, PrintNum, (void*)&thr_param[i]);
		}
	}
	for (int i=0; i<4; i++)
		pthread_join(thr[i], NULL);
}	

int main() {
	try2();

	return 0;
}


