echo compiling &&
nvcc kernel.cu -o fractal \
"-I/home/ionut/Downloads/SFML-3.0 (2).2-linux-gcc-64-bit/SFML-3.0.2/include" \
"-L/home/ionut/Downloads/SFML-3.0 (2).2-linux-gcc-64-bit/SFML-3.0.2/lib" \
-lsfml-graphics -lsfml-window -lsfml-system
