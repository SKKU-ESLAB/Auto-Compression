#include <pthread.h>
#include <iostream>

#define num_ch			(4)
#define num_thr_per_ch	(4)
#define num_iter_per_ch	(3) 

typedef struct {
	int ch;
	int num;
} thr_param_t;

int err = 0;
int prev[num_ch];

pthread_t thr[num_ch*num_thr_per_ch];
pthread_t thr_grp[num_ch];
thr_param_t thr_param[num_ch*num_thr_per_ch];
thr_param_t thr_grp_param[num_ch];
pthread_barrier_t thr_barrier[num_ch];
pthread_mutex_t mutex[num_ch];
pthread_mutex_t print_mutex;

static void* PrintNum(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	int num = input->num;

	pthread_mutex_lock(&print_mutex);
	for (int i=0; i<ch; i++)
		std::cout << "\t";
	std::cout << "   " << num/num_thr_per_ch << " " << num%num_thr_per_ch << "\n";

	err += num/num_thr_per_ch - prev[ch];
	prev[ch] = num/num_thr_per_ch;

	pthread_mutex_unlock(&print_mutex);
	pthread_barrier_wait(&thr_barrier[ch]);
	return (NULL);
}

static void* PrintNumChannel(void *input_) {
	thr_param_t *input = (thr_param_t*)input_;
	int ch = input->ch;
	int num = input->num;

	for (int off_i=0; off_i < num_iter_per_ch; off_i++) {
		pthread_barrier_init(&thr_barrier[ch], NULL, num_thr_per_ch+1);
		for (int off_f=0; off_f < num_thr_per_ch; off_f++) {
			int offset = off_i*num_thr_per_ch + off_f;
			thr_param[ch*num_thr_per_ch+off_f].ch = ch;
			thr_param[ch*num_thr_per_ch+off_f].num = offset;
			pthread_create(&thr[ch*num_thr_per_ch+off_f], NULL, PrintNum, (void*)&thr_param[ch*num_thr_per_ch+off_f]);
		}
		pthread_barrier_wait(&thr_barrier[ch]);
	}
	return (NULL);
}

void Process() {
	for (int ch=0; ch < num_ch; ch++) {
		thr_grp_param[ch].ch = ch;
		thr_grp_param[ch].num = 0;
		pthread_create(&thr_grp[ch], NULL, PrintNumChannel, (void*)&thr_grp_param[ch]);
	}
}

int main() {
	for (int i=0; i<num_ch; i++)
		std::cout << "   ch" << i << "\t";
	std::cout << "\n";
	
	Process();
		
	for (int ch=0; ch < num_ch; ch++) {
		pthread_join(thr_grp[ch], NULL);
		for (int offset=0; offset < num_thr_per_ch; offset++)
			pthread_join(thr[ch*num_thr_per_ch+offset], NULL);
	}

	std::cout << "\nfinished!\n";
	std::cout << "err : " << err - num_ch*(num_iter_per_ch-1) << "\n";
	return 0;
}	

