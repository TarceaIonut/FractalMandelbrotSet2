#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>

#include <stdio.h>
#include <iostream>
#include <chrono>

#define HEIGHT 1080
#define WIDTH 1920

#define P_TIME 0x1
#define P_COORDS 0x2
#define P_GPU 0x4

#define PROCENT_DIRECTION_MOVEMENT 0.05f

#define DEFAULT_PRINT P_TIME | P_GPU

#define V_ZOOM 0.1f
#define DISTANCE_INIT_X 2.0f
#define POZ_INIT_X -1.0f
#define POZ_INIT_Y -1.0f

struct set_color_info {
    int* a;
    int nr;
    double start_x, start_y, end_x, end_y;
};
struct running_info {
    int print = DEFAULT_PRINT;
    bool test_GPU = true;
    int threadsPerBlock = 256;
    int* gpu_reserved_memory;
};
bool init_running_info(running_info& info) {
    auto e = cudaMalloc((void**)&info.gpu_reserved_memory, HEIGHT * WIDTH * sizeof(int));
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(info.gpu_reserved_memory);
        return false;
    }
    return true;
}
int init_set_color_info(set_color_info& info) {
    info.a = (int*)malloc(HEIGHT * WIDTH * sizeof(int));
    info.nr = 1024;
    info.start_x = POZ_INIT_X;
    info.start_y = POZ_INIT_Y;
    info.end_x = DISTANCE_INIT_X + POZ_INIT_X;
    info.end_y = POZ_INIT_Y + DISTANCE_INIT_X;
    return 0;
}

__global__ void set_number_for_color_cuda(set_color_info info);

cudaError_t set_number_for_color(set_color_info& info, running_info &r_info);
cudaError_t set_image_GUP(set_color_info& info, running_info& r_info, sf::Image& image, double *gpu_calculation_time, double* set_pixels_time);

cudaError_t display_GPU();

cudaError_t set_number_for_color(set_color_info &info, running_info &r_info) {
    cudaError_t cudaStatus;
    int threadsPerBlock = r_info.threadsPerBlock;
    int size = HEIGHT * WIDTH;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    set_color_info cuda_info = info;
    cuda_info.a = r_info.gpu_reserved_memory;
    set_number_for_color_cuda << <blocksPerGrid, threadsPerBlock >> > (cuda_info);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching set_number_for_color_cuda!\n", cudaStatus);
        goto Error;
    }
    cudaMemcpy(info.a, cuda_info.a, size * sizeof(int), cudaMemcpyDeviceToHost);
Error:
    //cudaFree(cuda_info.a);
    return cudaStatus;
}
cudaError_t set_image_GUP(set_color_info& info, running_info& r_info, sf::Image& image, double *gpu_calculation_time, double* set_pixels_time) {
    auto start = std::chrono::high_resolution_clock::now();
    auto err = set_number_for_color(info, r_info);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    int pixel_nr = 0;
    for (int x = 0; x < HEIGHT; x++) {
        for (int y = 0; y < WIDTH; y++) {
            int normalized_nr = info.a[pixel_nr++];
            if (info.nr > 256) normalized_nr = normalized_nr / (info.nr / 256);
            else if (info.nr < 256) normalized_nr *= (256 / info.nr);
            image.setPixel(sf::Vector2u( y, x ), sf::Color(normalized_nr, 0, 256 - normalized_nr));
        }
    }
    auto end_setPixel = std::chrono::high_resolution_clock::now();
    if (gpu_calculation_time != NULL) {
        *gpu_calculation_time = std::chrono::duration<double, std::milli>(end_gpu - start).count() / 1000.0f;
    }
    if (set_pixels_time != NULL) {
        *set_pixels_time = std::chrono::duration<double, std::milli>(end_setPixel - end_gpu).count() / 1000;
    }
    return err;
}
__global__ void set_number_for_color_cuda(set_color_info info) {
    int number = blockIdx.x * blockDim.x + threadIdx.x;
    if (number >= HEIGHT * WIDTH) {
        return;
    }
    int x = number / WIDTH;
    int y = number % WIDTH;
    double imag = ((double)x / HEIGHT) * (info.end_x - info.start_x) + info.start_x;
    double real = ((double)y / WIDTH) * (info.end_y - info.start_y) + info.start_y;
    double r = 0;
    double im = 0;
    for (int k = 0; k < info.nr; k++) {
        double x = r * r - im * im + real;
        double y = 2 * r * im + imag;
        r = x;
        im = y;
        if (r * r + y * y > (double)1'000'000'000'000.0) {
            info.a[number] = k;
            return;
        }
    }
    info.a[number] = info.nr / 2;
}

void mouseCoordsToRealValues(int mouse_x, int mouse_y, set_color_info& color_info, double& poz_x, double& poz_y);
void setCenerToMousePoz(int mouse_x, int mouse_y, set_color_info& color_info, double zoom = 1);
void centerOnRealPosition(set_color_info& color_info, double poz_x, double poz_y, double zoom = 1);
void zoomInplace(set_color_info& color_info, double zoom);
void zoomOnCoords(set_color_info& color_info, int mouse_x, int mouse_y, double zoom);
void modeToDirection(set_color_info& color_info, double direction_p_x, double direction_p_y);

int main(){
    sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Fractal", sf::State::Fullscreen);
    sf::Image image(sf::Vector2u(WIDTH, HEIGHT), sf::Color::Black);

    set_color_info color_info;
    running_info running_info;

    init_running_info(running_info);
    init_set_color_info(color_info);

    sf::Texture texture;
    bool texture_loaded = texture.loadFromImage(image);
    if (!texture_loaded) { std::cout << "Error loading texture from image\n"; return -1;}
    sf::Sprite sprite(texture);
    if (running_info.print & P_GPU){ display_GPU(); }
    while (window.isOpen()) {
        while (const std::optional event_ = window.pollEvent()) {
            const sf::Event& e = *event_;
            if (e.is<sf::Event::Closed>()) {
                window.close();
            }
            if (const auto* keyPressed = e.getIf<sf::Event::KeyPressed>()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) {
                    window.close();
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) {
                    zoomInplace(color_info, 2);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Backspace)) {
                    zoomInplace(color_info, 0.5);
                }
                if (keyPressed->code == sf::Keyboard::Key::W || keyPressed->code == sf::Keyboard::Key::Up) {
                    modeToDirection(color_info, 0.5 - PROCENT_DIRECTION_MOVEMENT, 0.5);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left)) {
                    modeToDirection(color_info, 0.5, 0.5 - PROCENT_DIRECTION_MOVEMENT);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down)) {
                    modeToDirection(color_info, 0.5 + PROCENT_DIRECTION_MOVEMENT, 0.5);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right)) {
                    modeToDirection(color_info, 0.5, 0.5 + PROCENT_DIRECTION_MOVEMENT);
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::N)) {
                    color_info.nr *= 2;
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::M)) {
                    if (color_info.nr >= 32)
                        color_info.nr /= 2;
                }
            }
            if (const auto* mouseClick = e.getIf<sf::Event::MouseButtonPressed>()) {
                sf::Vector2i v = mouseClick->position;
                if (running_info.print & P_COORDS) {
                    std::cout << "mouse position = " << v.x << ' ' << v.y << '\n';
                }
                setCenerToMousePoz(v.y, v.x, color_info);
            }
            if (const auto* scroll = e.getIf<sf::Event::MouseWheelScrolled>()) {
                sf::Vector2i v = sf::Mouse::getPosition();
                int wheel = scroll->delta;
                if (running_info.print & P_COORDS) {
                    std::cout << "mouse position = " << v.x << ' ' << v.y << '\n';
                }
                if (wheel == 1) {
                    zoomOnCoords(color_info, v.y, v.x, 1.1);
                }
                else {
                    zoomOnCoords(color_info, v.y, v.x, 0.91);
                }
            }
        }

        if (!window.isOpen()) { continue; }

        double gpu_time, set_pixels_time;
        set_image_GUP(color_info, running_info, image, &gpu_time, &set_pixels_time);

        auto start = std::chrono::high_resolution_clock::now();

        bool texture_loaded = texture.loadFromImage(image);
        sprite.setTexture(texture);
        window.clear();
        window.draw(sprite);
        window.display();

        auto end = std::chrono::high_resolution_clock::now();

        if (running_info.print & P_TIME && texture_loaded) {
            printf("gpu_calculation_time = %lf, set_pixels_time = %lf, display_time = %lf\n", gpu_time, set_pixels_time, 
                std::chrono::duration<double, std::milli>(end - start).count() / 1000);
        }
    }
    return 0;
}
cudaError_t display_GPU() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    return cudaStatus;
}
void setCoordonates(double poz1, double poz2, double p, set_color_info& color_info) {
    color_info.start_x = poz1 - p;
    color_info.end_x = poz1 + p;
    color_info.start_y = poz2 - p;
    color_info.end_y = poz2 + p;
}
void mouseCoordsToRealValues(int mouse_x, int mouse_y, set_color_info& color_info, double& poz_x, double& poz_y) {
    double procent_x = (double)mouse_x / HEIGHT;
    double procent_y = (double)mouse_y / WIDTH;

    poz_x = (color_info.end_x - color_info.start_x) * procent_x + color_info.start_x;
    poz_y = (color_info.end_y - color_info.start_y) * procent_y + color_info.start_y;
}
void centerOnRealPosition(set_color_info& color_info, double poz_x, double poz_y, double zoom) {
    double d_x = (color_info.end_x - color_info.start_x) * zoom / 2;
    double d_y = (color_info.end_y - color_info.start_y) * zoom / 2;

    color_info.start_x = poz_x - d_x;
    color_info.end_x = poz_x + d_x;

    color_info.start_y = poz_y - d_y;
    color_info.end_y = poz_y + d_y;
}
void zoomInplace(set_color_info& color_info, double zoom) {
    double d_x = (color_info.end_x - color_info.start_x) * (zoom - 1) / 2;
    double d_y = (color_info.end_y - color_info.start_y) * (zoom - 1) / 2;

    color_info.start_x -= d_x;
    color_info.end_x += d_x;
    color_info.start_y -= d_y;
    color_info.end_y += d_y;
}
void zoomOnCoords(set_color_info& color_info, int mouse_x, int mouse_y, double zoom) {
    double procent_x = (double)mouse_x / HEIGHT;
    double procent_y = (double)mouse_y / WIDTH;

    double d_x = (color_info.end_x - color_info.start_x) * (zoom - 1);
    double d_y = (color_info.end_y - color_info.start_y) * (zoom - 1);

    color_info.start_x -= d_x * procent_x;
    color_info.end_x += d_x * (1 - procent_x);
    color_info.start_y -= d_y * procent_y;
    color_info.end_y += d_y * (1 - procent_y);

}
void setCenerToMousePoz(int mouse_x, int mouse_y, set_color_info& color_info, double zoom) {
    double poz_x, poz_y;
    mouseCoordsToRealValues(mouse_x, mouse_y, color_info, poz_x, poz_y);
    centerOnRealPosition(color_info, poz_x, poz_y);
}
void modeToDirection(set_color_info& color_info, double direction_p_x, double direction_p_y) {
    double poz_x, poz_y;
    mouseCoordsToRealValues(direction_p_x * HEIGHT, direction_p_y * WIDTH, color_info, poz_x, poz_y);
    centerOnRealPosition(color_info, poz_x, poz_y);
}