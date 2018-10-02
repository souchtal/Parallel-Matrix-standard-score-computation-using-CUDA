Compile code using

	nvcc -o cuda cudaParallel.cu

Submit job using

	sbatch runcuda.sh


Compile code using 

	gcc -o norm cudaSeq.c -lm

Submit job using

	sbatch runseq.sh
