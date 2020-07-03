all: main

main: main.cu
	/usr/local/cuda/bin/nvcc -lineinfo -Xcompiler -fopenmp -o $@ $^

run: main
	./main

clean:
	rm -rf main
