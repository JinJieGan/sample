#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>
using namespace std;

static inline bool within_bounds_2d(int h, int w, int H, int W)
{
  return h >= 0 && h < H && w >= 0 && w < W;
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

vector<vector<float>> grid_simple(vector<vector<float>> input,
                                  vector<vector<vector<float>>> grid,
                                  vector<vector<float>> output)
{
  Timer timer;
  int IH = input.size();
  int IW = input[0].size();
  int H = grid.size();
  int W = grid[0].size();
  float zero_point = 0.;
  double kernel_times = 0;
  float iters = 5;
  for (int i = 0; i < iters; i++)
  {
    // loop over each output pixel
    int h, w;
    auto start1 = timer.Elapsed();

    for (h = 0; h < H; ++h)
    {
      for (w = 0; w < W; ++w)
      {
        // get the corresponding input x, y co-ordinates from grid

        float rx = grid[h][w][0];
        float ry = grid[h][w][1];

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
                   ? input[iy_nw][ix_nw] * nw
                   : zero_point * nw;
        res += within_bounds_2d(iy_ne, ix_ne, IH, IW)
                   ? input[iy_ne][ix_ne] * ne
                   : zero_point * ne;
        res += within_bounds_2d(iy_sw, ix_sw, IH, IW)
                   ? input[iy_sw][ix_sw] * sw
                   : zero_point * sw;
        res += within_bounds_2d(iy_se, ix_se, IH, IW)
                   ? input[iy_se][ix_se] * se
                   : zero_point * se;
        output[h][w] = res; // std::round(res);
      }
    }
    auto end1 = timer.Elapsed();
    float kernel_time = (end1 - start1) * 1000.f;
    std::cerr << "kernel time=" << kernel_time << " msec\n";
    kernel_times += kernel_time;
  }
  std::cerr << "average kernel time=" << kernel_times/iters << " msec\n";
  return output;
}

int main()
{
  cout << "begin" << endl;
  int in_H = 512;
  int in_W = 960;
  int out_H = 2048;
  int out_W = 3840;
  vector<vector<float>> input(in_H, vector<float>(in_W, 0.0));
  vector<vector<vector<float>>> grid(out_H, vector<vector<float>>(out_W, vector<float>(2, 0.0000)));
  vector<vector<float>> output(out_H, vector<float>(out_W, 0.0));
  float x = 1;
  float y = 2;
  for (int j = 0; j < out_H; ++j)
  {
    for (int k = 0; k < out_W; ++k)
    {
      grid[j][k][0] = x;
      grid[j][k][1] = y;
    }
  }

  float pixel = 1.0;

  for (int k = 0; k < in_H; ++k)
  {
    for (int l = 0; l < in_W; ++l)
    {
      input[k][l] = pixel;
    }
  }

  Timer timer;
  auto start = timer.Elapsed();
  output = grid_simple(input, grid, output);
  auto end = timer.Elapsed();
  float total_time = (end - start) * 1000.f;
  std::cerr << "sample time=" << total_time << " msec\n";
}
