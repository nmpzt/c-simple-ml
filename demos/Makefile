CC = gcc
CFLAGS = -O

SRCS = $(wildcard *.c)

PROGS = $(SRCS:.c=)

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ -lm -Wall

clean:
	rm -f *.o *.obj *.exe *.out app demo
	rm *.ppm
	rm ./models/*.bin
