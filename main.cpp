
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <chrono>
#include "SFML\Graphics.hpp"
#include <thread>

//double p1 = -0.7361705050500805;
//double p2 = 0.14491345454543447;
//double des1 = -0.7361705050500401;
//double des2 = 0.14491345454547488;

double p1 = -1;
double p2 = -1;
double des1 = 1;
double des2 = 1;
int width = 1000;
int height = 1000;
int nr = 512;
int t = 12;
int** matrix;


struct complex {
	double real = 0;
	double imag = 0;
	inline complex operator+(complex other) {
		return { real + other.real, imag + other.imag };
	}
	inline complex operator*(complex other) {
		return { real * other.real - imag * other.imag, real * other.imag + imag * other.real };
	}
	inline double size() {
		return sqrt(real * real + imag * imag);
	}
};
struct complex_v2 {
	double real1 = 0, real2 = 0;
	double imag1 = 0, imag2 = 0;
	inline complex_v2 operator+(complex_v2 other) {
		return { real1 + other.real1, real2 + other.real2, imag1 + other.imag1, imag2 + other.imag2 };
	}
	inline complex_v2 operator*(complex_v2 other) {

		//return { real * other.real - imag * other.imag, real * other.imag + imag * other.real };
	}
};
int check(complex c) {
	complex z = { 0,0 };
	for (int i = 0; i < nr; i++) {
		z = z * z + c;
		if (z.real * z.real + z.imag * z.imag > (double)1'000'000'000.0) return i;
	}
	return nr / 2;
}
int check_v2(double real, double imag) {
	double r = 0;
	double im = 0;
	for (int i = 0; i < nr; i++) {
		double x = r * r - im * im + real;
		double y = 2 * r * im + imag;
		r = x;
		im = y;
		if (r * r + y * y > (double)1'000'000'000.0) {
			return i;
		}
	}
	return nr / 2;
}
int check_until_end(int i, int j, complex z, int k) {
	complex c;
	c.real = (double)i / height * (des1 - p1) + p1;
	c.imag = (double)j / width * (des2 - p2) + p2;
	for (int i = k; i < nr; i++) {
		z = z * z + c;
		if (z.size() > 10000) return i;
	}
	return 0;
}
void check_smart(int i, int j, complex z, int k) {
	if (k == nr) {
		matrix[i][j] = 0;
	}
	else if (matrix[i][j] == -1) {
		complex c;
		c.real = (double)i / height * (des1 - p1) + p1;
		c.imag = (double)j / width * (des2 - p2) + p2;
		z = z * z + c;
		int I = (z.real - p1) * height / (des1 - p1);
		int J = (z.imag - p2) * width / (des2 - p2);
		c.real = (double)I / height * (des1 - p1) + p1;
		c.imag = (double)J / width * (des2 - p2) + p2;
		//std::cout << z.real << ' ' << z.imag << ' ' << c.real << ' ' << c.imag << '\n';
		int n;
		if (I < 0 || J < 0 || I >= height || J >= width) {
			n = check_until_end(I, J, z, k + 1);
			//std::cout << z.real << ' ' << z.imag << ' ' << c.real << ' ' << c.imag << '\n';
			//std::cout << "I = " << I << " J = " << J << " k = " << k << " n = " << n << '\n';
			if (n == 0) {
				//std::cout << "I = " << I << " J = " << J << " k = " << k << '\n';
			}
		}
		else {
			check_smart(I, J, z, k + 1);
			n = matrix[I][J];
		}
		if (n == 0) {
			matrix[i][j] = 0;
		}
		else {
			matrix[i][j] = n + 1;
		}
	}
}
void zoom(double poz1, double poz2, double level) {
	double p = (((des1 - p1) / (2 * level)));
	std::cout << "p1 = " << p1 << " p2 = " << p2 << " des1 = " << des1 << " des2 =  " << des2 << "p = " << p << '\n';
	poz1 = (des1 - p1) / 10 * poz1 + p1;
	poz2 = (des2 - p2) / 10 * poz2 + p2;
	p1 = poz1 - p;
	des1 = poz1 + p;
	p2 = poz2 - p;
	des2 = poz2 + p;
	std::cout << "p1 = " << p1 << " p2 = " << p2 << " des1 = " << des1 << " des2 =  " << des2 << "p = " << p << '\n';
}
void setCoordonates(double poz1, double poz2, double p) {
	p1 = poz1 - p;
	des1 = poz1 + p;
	p2 = poz2 - p;
	des2 = poz2 + p;
}
void make_image(sf::Image *image, int startH = 0, int startW = 0, int h = height, int w = width) {
	int Nr;
	complex c;
	for (int i = startH; i < h; i++) {
		for (int j = startW; j < w; j++) {
			c.real = (double)i / height * (des1 - p1) + p1;
			c.imag = (double)j / width * (des2 - p2) + p2;
			Nr = check(c);
			if (nr > 256) Nr = Nr / (nr / 256);
			else if (nr < 256) Nr *= (256 / nr);
			(*image).setPixel(j, i, sf::Color(Nr, 0, 256 - Nr));
		}
	}
}
void make_image_v2(sf::Image* image, int X) {
	int Nr;
	complex c;
	for (int i = 0; i < height; i++) {
		for (int j = X; j < width; j+=t) {
			c.real = (double)i / height * (des1 - p1) + p1;
			c.imag = (double)j / width * (des2 - p2) + p2;
			Nr = check(c);
			if (nr > 256) Nr = Nr / (nr / 256);
			else if (nr < 256) Nr *= (256 / nr);
			(*image).setPixel(j, i, sf::Color(Nr, 0, 256 - Nr));
		}
	}
}
void make_image_v3(sf::Image* image, int X) {
	int Nr;
	complex c;
	for (int i = 0; i < height; i++) {
		for (int j = X; j < width; j ++) {
			check_smart(i, j, {0, 0}, 0);
			if (matrix[i][j] == 0) Nr = nr / 2;
			else Nr = matrix[i][j];
			if (nr > 256) Nr = Nr / (nr / 256);
			else if (nr < 256) Nr *= (256 / nr);
			(*image).setPixel(j, i, sf::Color(Nr, 0, 256 - Nr));
		}
	}
}
void display_sng(sf::Image* image, sf::Sprite* sprite, sf::Texture* texture, sf::RenderWindow* window) {
	auto start = std::chrono::high_resolution_clock::now();
	make_image(image);
	auto end1 = std::chrono::high_resolution_clock::now();
	double duration1 = std::chrono::duration<double, std::milli>(end1 - start).count();
	std::cout << '\n' << duration1 / 1000 << '\n';
	(*texture).loadFromImage(*image);  //Load Texture from image
	(*sprite).setTexture(*texture);
	(*window).draw(*sprite);
	(*window).display();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << '\n' << duration / 1000 << '\n';
}
void display_multi_th(sf::Image* image, sf::Sprite* sprite, sf::Texture* texture, sf::RenderWindow* window) {
	std::vector<std::thread> threads;
	auto start = std::chrono::high_resolution_clock::now();
	int amountH, pozH = 0;
	for (int i = 0; i < t; i++) {
		amountH = (height - pozH) / (t - i);
		threads.emplace_back(make_image, image, pozH, 0, pozH + amountH, width);
		pozH += amountH;
	}
	for (std::thread& t : threads) {
		t.join();
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	double duration1 = std::chrono::duration<double, std::milli>(end1 - start).count();
	(*texture).loadFromImage(*image);  //Load Texture from image
	(*sprite).setTexture(*texture);
	(*window).draw(*sprite);
	(*window).display();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << '\n' << duration / 1000 << '\n';
}
void display_multi_th_v2(sf::Image* image, sf::Sprite* sprite, sf::Texture* texture, sf::RenderWindow* window) {
	std::vector<std::thread> threads;
	auto start = std::chrono::high_resolution_clock::now();
	int amountH, pozH = 0;
	for (int i = 0; i < t; i++) {
		threads.emplace_back(make_image_v2, image, i);
	}
	for (std::thread& t : threads) {
		t.join();
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	double duration1 = std::chrono::duration<double, std::milli>(end1 - start).count();
	(*texture).loadFromImage(*image);  //Load Texture from image
	(*sprite).setTexture(*texture);
	(*window).draw(*sprite);
	(*window).display();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << '\n' << duration / 1000 << '\n';
}
void display_smart(sf::Image* image, sf::Sprite* sprite, sf::Texture* texture, sf::RenderWindow* window) {
	auto start = std::chrono::high_resolution_clock::now();
	make_image_v3(image, 0);
	for (int i = 0; i < height; i++) {
		memset(matrix[i], -1, width);
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	double duration1 = std::chrono::duration<double, std::milli>(end1 - start).count();
	std::cout << '\n' << duration1 / 1000 << '\n';
	(*texture).loadFromImage(*image);  //Load Texture from image
	(*sprite).setTexture(*texture);
	(*window).draw(*sprite);
	(*window).display();
	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << '\n' << duration / 1000 << '\n';
}

int main() {
	matrix = (int**)malloc(sizeof(int*) * height);
	for (int i = 0; i < height; i++) {
		matrix[i] = (int*)malloc(sizeof(int) * width);
		memset(matrix[i], -1, width);
	}
	sf::RenderWindow window(sf::VideoMode(width, height), "Window Title");
	sf::Image image;
	sf::Sprite sprite;
	sf::Texture texture;
	sf::Event event;
	image.create(height, width, sf::Color::Black);
	double poz1, poz2, level;
	display_multi_th(&image, &sprite, &texture, &window);

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			if ((event.type == sf::Event::Closed) || (event.type == sf::Event::KeyPressed)) {
				if (event.key.code == sf::Keyboard::Space) {
					zoom(5, 5, 0.5);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::Backspace) {
					zoom(5, 5, 2);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::W) {
					zoom(4, 5, 1);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::S) {
					zoom(6, 5, 1);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::A) {
					zoom(5, 4, 1);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::D) {
					zoom(5, 6, 1);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::N) {
					nr *= 2;
					std::cout << "nr = " << nr << '\n';
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::M) {
					nr /= 2;
					std::cout << "nr = " << nr << '\n';
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::T) {
					t++;
					std::cout << "t = " << t << '\n';
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::Y) {
					if (t > 1) {
						t--;
						std::cout << t << '\n';
					}
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::R) {
					setCoordonates(0, 0, 1);
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::F) {
					height = 1080;
					width = 1920;
					display_multi_th_v2(&image, &sprite, &texture, &window);
				}
				else if (event.key.code == sf::Keyboard::Escape) {
					return 0;
				}
			}
		}
	}
	return 0;
}