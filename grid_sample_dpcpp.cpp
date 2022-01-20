#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <math.h>
#include <chrono>

using namespace std;
using namespace cl::sycl;
constexpr int IH = 512;
constexpr int IW = 960;
constexpr int OH = 2048;
constexpr int OW = 3840;

static inline bool within_bounds_2d(int h, int w, int H, int W)
{
    return h >= 0 && h < H&& w >= 0 && w < W;
}

class Timer
{
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double Elapsed()
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Duration>(now - start_).count();
    }

private:
    using Duration = std::chrono::duration<double>;
    std::chrono::steady_clock::time_point start_;
};

auto grid_sample(float(*input)[IW], float(*grid)[OW][2], sycl::queue q)
{
    Timer timer;
    float zero_point = 0.;
    float(*output)[OW] = new float[OH][OW];
    try
    {

        buffer input_buf(reinterpret_cast<float*>(input), range(IH, IW));
        buffer grid_buf(reinterpret_cast<float*>(grid), range(OH, OW, 2));
        buffer output_buf(reinterpret_cast<float*>(output), range(OH, OW));

        double etime = 0;
        double start;
        double kernel_times = 0;
        float iters = 5;
        // loop over each output pixel
        for (int i = 0; i < iters; i++)
        {
            double start0 = timer.Elapsed();
            q.submit([&](auto& h)
                {
                    // Read from input and grid, write to output
                    accessor inp(input_buf, h, read_only);
                    accessor gri(grid_buf, h, read_only);
                    accessor out(output_buf, h, write_only);
                    // Execute kernel.
                    h.parallel_for(range(OH, OW), [=](auto index)
                        {
                            // Get global position in Y direction.
                            int h = index[0];
                            // Get global position in X direction.
                            int w = index[1];
                            // get the corresponding input x, y co-ordinates from grid
                            float rx = gri[h][w][0];
                            float ry = gri[h][w][1];
                            // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
                            float ix = ((rx + 1) / 2) * (IW - 1);
                            float iy = ((ry + 1) / 2) * (IH - 1);
                            // get NE, NW, SE, SW pixel values from (x, y)
                            int ix_nw = std::floor(ix);
                            int iy_nw = std::floor(iy);
                            int ix_ne = ix_nw + 1;
                            int iy_ne = iy_nw;
                            int ix_sw = ix_nw;
                            int iy_sw = iy_nw + 1;
                            int ix_se = ix_nw + 1;
                            int iy_se = iy_nw + 1;
                            // get surfaces to each neighbor:
                            float nw = (ix_se - ix) * (iy_se - iy);
                            float ne = (ix - ix_sw) * (iy_sw - iy);
                            float sw = (ix_ne - ix) * (iy - iy_ne);
                            float se = (ix - ix_nw) * (iy - iy_nw);
                            // calculate bilinear weighted pixel value and set output pixel

                            //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
                            // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
                            float res = 0;
                            res += within_bounds_2d(iy_nw, ix_nw, IH, IW)
                                ? inp[iy_nw][ix_nw] * nw
                                : zero_point * nw;
                            res += within_bounds_2d(iy_ne, ix_ne, IH, IW)
                                ? inp[iy_ne][ix_ne] * ne
                                : zero_point * ne;
                            res += within_bounds_2d(iy_sw, ix_sw, IH, IW)
                                ? inp[iy_sw][ix_sw] * sw
                                : zero_point * sw;
                            res += within_bounds_2d(iy_se, ix_se, IH, IW)
                                ? inp[iy_se][ix_se] * se
                                : zero_point * se;
                            out[h][w] = res; // std::round(res);

                        });
                });
            double end0 = timer.Elapsed();
            etime = (end0 - start0) * 1000.f;
            std::cerr << " kernel time=" << etime << " msec\n";
            if (i > 0)
                kernel_times += etime;
            else
                start = timer.Elapsed();
        }
        double end = timer.Elapsed();
        float total_time = (end - start) * 1000.f / iters;
        float kernel_time = kernel_times / iters;
        std::cerr << "average kernel time = " << kernel_time << " msec\n";
        std::cerr << "total time = " << total_time << " msec\n";
    }
    catch (sycl::exception const& e)
    {
        cout << "An exception is caught while sampling.\n";
        terminate();
    }

    return output;
}

int main()
{
    float(*input)[IW] = new float[IH][IW];
    float(*grid)[OW][2] = new float[OH][OW][2];
    for (int j = 0; j < OH; ++j)
    {
        for (int k = 0; k < OW; ++k)
        {
            grid[j][k][0] = 1.0f;
            grid[j][k][1] = 2.0f;
        }
    }

    float pixel = 1.0;
    for (int k = 0; k < IH; ++k)
    {
        for (int l = 0; l < IW; ++l)
        {
            input[k][l] = pixel;
            pixel = 1.0;
        }
    }

    Timer timer;
    sycl::queue q(cpu_selector{});
    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
    auto start = timer.Elapsed();
    float(*output)[OW] = grid_sample(input, grid, q);
    auto end = timer.Elapsed();
    float total_time = (end - start) * 1000.f;
    std::cerr << "sample time=" << total_time << " msec\n";
    delete[] output;
}
