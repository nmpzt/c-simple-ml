CC = gcc
CFLAGS = -O

SRCS = $(wildcard *.c)

PROGS = $(SRCS:.c=)

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(PROGS) *.exe
	rm output.ppm
	rm ./models/xor_model.bin
	rm ./models/spiral_model.bin
