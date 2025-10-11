// CSCI 5607 HW 2 - Image Conversion Instructor: S. J. Guy <sjguy@umn.edu>
// In this assignment you will load and convert between various image formats.
// Additionally, you will manipulate the stored image data by quantizing,
// cropping, and suppressing channels

#include "image.h"
#include "pixel.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <string.h>

#include <fstream>
using namespace std;

int map_to_midbucket(int value, int levels) {
  return ((2 * value + 1) * 255 + levels) / (2 * levels);
}
// TODO - HW2: The current implementation of read_ppm() assumes the PPM file has
// a maximum value of 255 (ie., an 8-bit PPM) ...
// TODO - HW2: ... you need to adjust the function to support PPM files with a
// max value of 1, 3, 7, 15, 31, 63, 127, and 255 (why these numbers?)
uint8_t *read_ppm(char *imgName, int &width, int &height) {
  // Open the texture image file
  ifstream ppmFile;
  ppmFile.open(imgName);
  if (!ppmFile) {
    printf("ERROR: Image file '%s' not found.\n", imgName);
    exit(1);
  }

  // Check that this is an ASCII PPM (first line is P3)
  string PPM_style;
  ppmFile >> PPM_style; // Read the first line of the header
  if (PPM_style != "P3") {
    printf("ERROR: PPM Type number is %s. Not an ASCII (P3) PPM file!\n",
           PPM_style.c_str());
    exit(1);
  }

  // Read in the texture width and height
  ppmFile >> width >> height;
  unsigned char *img_data = new unsigned char[4 * width * height];

  // Check that the 3rd line is 255 (ie., this is an 8 bit/pixel PPM)
  int maximum;
  ppmFile >> maximum;
  int bits = log2(maximum + 1);
  // TODO - HW2: Remove this check below, instead make the function for for
  // maximum values besides 255
  // if (maximum != 255) {
  //   printf("ERROR: We assume Maximum size is 255, not (%d)\n", maximum);
  //   printf("TODO: This error means you didn't finish your HW yet!\n");
  //   exit(1);
  // }

  // TODO - HW2: The values read from the file might not be 8-bits (ie, a
  // maximum values besides 255)
  // TODO - HW2: However img_data stores all values as 8-bit integers.
  // TODO - HW2: When you read the values into img_data scale the values up to
  // be 8 bits
  // TODO - HW2: For example, the value 1 in a 1 bit PPM should become 255 ...
  // TODO - HW2: Likewise, the value 1 in a 2 bit PPM should become 127 (or
  // 128).
  int r, g, b;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ppmFile >> r >> g >> b;
      int idx = i * width * 4 + j * 4;
      img_data[idx] = map_to_midbucket(r, maximum + 1);
      img_data[idx + 1] = map_to_midbucket(g, maximum + 1);
      img_data[idx + 2] = map_to_midbucket(b, maximum + 1);
      img_data[idx + 3] = 255; // Alpha
    }
  }

  return img_data;
}

int map_from_midbucket(int value, int levels) { return (value * levels) / 256; }

void write_ppm(char *imgName, int width, int height, int bits,
               const uint8_t *data) {
  // Open the texture image file
  ofstream ppmFile;
  ppmFile.open(imgName);
  if (!ppmFile) {
    printf("ERROR: Could not create file '%s'\n", imgName);
    exit(1);
  }

  // Set this as an ASCII PPM (first line is P3)
  string PPM_style = "P3\n";
  ppmFile << PPM_style; // Read the first line of the header

  // Write out the texture width and height
  ppmFile << width << " " << height << "\n";

  // Set's the 3rd line to 255 (ie., assumes this is an 8 bit/pixel PPM)
  // TODO - HW2: Set the maximum values based on the value of the variable
  // 'bits'
  int maximum = (1 << bits) - 1;
  ppmFile << maximum << "\n";

  // TODO - HW2: The values in data are all 8 bits, you must convert down to
  // whatever the variable bits is when writing the file
  //
  float scale = maximum / 255.0f;

  int r, g, b, a;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx = i * width * 4 + j * 4;

      r = map_from_midbucket(data[idx + 0], maximum + 1);
      g = map_from_midbucket(data[idx + 1], maximum + 1);
      b = map_from_midbucket(data[idx + 2], maximum + 1);
      // if (i == 0 && j < 10) {
      //   printf("r: %f, g: %f, b: %f", data[idx + 0] * scale,
      //          data[idx + 1] * scale, data[idx + 2] * scale);
      //   printf("r: %d, g: %d, b: %d\n", r, g, b);
      // }
      a = data[idx + 3]; // Alpha
      ppmFile << r << " " << g << " " << b << " ";
    }
  }

  ppmFile.close();
}

/**
 * Image
 **/
Image::Image(int width_, int height_) {

  assert(width_ > 0);
  assert(height_ > 0);

  width = width_;
  height = height_;
  num_pixels = width * height;
  sampling_method = IMAGE_SAMPLING_POINT;

  data.raw = new uint8_t[num_pixels * 4];
  int b = 0; // which byte to write to
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      data.raw[b++] = 0;
      data.raw[b++] = 0;
      data.raw[b++] = 0;
      data.raw[b++] = 0;
    }
  }

  assert(data.raw != NULL);
}

Image::Image(const Image &src) {
  width = src.width;
  height = src.height;
  num_pixels = width * height;
  sampling_method = IMAGE_SAMPLING_POINT;

  data.raw = new uint8_t[num_pixels * sizeof(Pixel)];

  memcpy(data.raw, src.data.raw, num_pixels * sizeof(Pixel));
}

Image::Image(char *fname) {

  int numComponents; //(e.g., Y, YA, RGB, or RGBA)

  // Load the pixels with STB Image Lib
  //
  int lastc = strlen(fname);
  uint8_t *loadedPixels;
  if (string(fname + lastc - 3) == "ppm") {
    loadedPixels = read_ppm(fname, width, height);
  } else {
    int numComponents; //(e.g., Y, YA, RGB, or RGBA)
    loadedPixels = stbi_load(fname, &width, &height, &numComponents, 4);
  }
  if (loadedPixels == NULL) {
    printf("Error loading image: %s", fname);
    exit(-1);
  }

  // Set image member variables
  num_pixels = width * height;
  sampling_method = IMAGE_SAMPLING_POINT;

  // Copy the loaded pixels into the image data structure
  data.raw = new uint8_t[num_pixels * sizeof(Pixel)];
  memcpy(data.raw, loadedPixels, num_pixels * sizeof(Pixel));
  free(loadedPixels);
}

Image::~Image() {
  delete[] data.raw;
  data.raw = NULL;
}

void Image::Write(char *fname) {

  int lastc = strlen(fname);

  switch (fname[lastc - 1]) {
  case 'm': // ppm
    write_ppm(fname, width, height, export_depth, data.raw);
    break;
  case 'g': // jpeg (or jpg) or png
    if (fname[lastc - 2] == 'p' || fname[lastc - 2] == 'e')  // jpeg or jpg
      stbi_write_jpg(fname, width, height, 4, data.raw, 95); // 95% jpeg quality
    else                                                     // png
      stbi_write_png(fname, width, height, 4, data.raw, width * 4);
    break;
  case 'a': // tga (targa)
    stbi_write_tga(fname, width, height, 4, data.raw);
    break;
  case 'p': // bmp
  default:
    stbi_write_bmp(fname, width, height, 4, data.raw);
  }
}

void Image::Brighten(double factor) {
  int x, y;
  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      Pixel scaled_p = p * factor;
      GetPixel(x, y) = scaled_p;
    }
  }
}

void Image::ExtractChannel(int channel) {
  int x, y;
  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      switch (channel) {
      case 0:
        p.g = 0;
        p.b = 0;
        break;
      case 1:
        p.r = 0;
        p.b = 0;
        break;
      case 2:
        p.r = 0;
        p.g = 0;
        break;
      }
      GetPixel(x, y) = p;
    }
  }
}

void Image::Quantize(int nbits) {
  for (int x = 0; x < Width(); x++) {
    for (int y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      Pixel new_p = PixelQuant(p, nbits);
      GetPixel(x, y) = new_p;
    }
  }
}

Image *Image::Crop(int x, int y, int w, int h) {
  Image *new_img = new Image(w, h);
  for (int i = x; i < x + w; i++) {
    for (int j = y; j < y + h; j++) {
      Pixel p = GetPixel(i, j);
      new_img->SetPixel(i - x, j - y, p);
    }
  }
  return new_img;
}

void Image::AddNoise(double factor) {
  int x, y;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255); //
  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      Pixel new_pixel = Pixel();
      new_pixel.SetClamp((int)((1 - factor) * p.r + factor * dist(rng)),
                         (int)((1 - factor) * p.g + factor * dist(rng)),
                         (int)((1 - factor) * p.b + factor * dist(rng)));
      GetPixel(x, y) = new_pixel;
    }
  }
}

// thanks to
// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
//
void Image::ChangeContrast(double factor) {
  int x, y;
  double avg = 0.0f;
  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      avg += 0.3 * p.r + 0.59 * p.g + 0.11 * p.b;
    }
  }
  avg /= num_pixels;
  // printf("avg: %f\n", avg);
  // printf("avg: %d\n", (int)avg);
  double f = (259 * (factor + 255)) / (255 * (259 - factor));
  // printf("f: %f\n", f);

  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      Pixel new_pixel = Pixel();
      new_pixel.SetClamp((int)(avg + f * (p.r - avg)),
                         (int)(avg + f * (p.g - avg)),
                         (int)(avg + f * (p.b - avg)));
      // printf("scale %d, %d %d\n", new_pixel.r, new_pixel.g, new_pixel.b);
      GetPixel(x, y) = new_pixel;
    }
  }
}

void Image::ChangeSaturation(double factor) {
  double f = (259 * (factor + 255)) / (255 * (259 - factor));
  int x, y;
  for (x = 0; x < Width(); x++) {
    for (y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      Pixel new_pixel = Pixel();
      float grayscale = 0.3 * p.r + 0.59 * p.g + 0.11 * p.b;
      new_pixel.SetClamp((int)(grayscale + f * (p.r - grayscale)),
                         (int)(grayscale + f * (p.g - grayscale)),
                         (int)(grayscale + f * (p.b - grayscale)));
      GetPixel(x, y) = new_pixel;
    }
  }
}

// For full credit, check that your dithers aren't making the pictures
// systematically brighter or darker
void Image::RandomDither(int nbits) {
  std::random_device dev;
  std::mt19937 rng(dev());

  this->export_depth = nbits;
  int maximum = (1 << nbits) - 1;
  double step = 255.0 / maximum;
  std::uniform_real_distribution<double> dist(-step/2.0f, step/2.0f);
  for (int x = 0; x < Width(); x++) {
    for (int y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);

      int r = p.r + dist(rng) + 0.5;
      int g = p.g + dist(rng) + 0.5;
      int b = p.b + dist(rng) + 0.5;

      double level_r = round((r * maximum) / 255.0f);
      double level_g = round((g * maximum) / 255.0f);
      double level_b = round((b * maximum) / 255.0f);

      // Pixel new_pixel = Pixel();
      // new_pixel.SetClamp(r, g, b);
      // GetPixel(x, y) = new_pixel;

      Pixel new_pixel = Pixel();
      new_pixel.SetClamp(level_r * step, level_g * step, level_b * step);
      this->SetPixel(x, y, new_pixel);
    }
  }
}
// This bayer method gives the quantization thresholds for an ordered dither.
// This is a 4x4 dither pattern, assumes the values are quantized to 16 levels.
// You can either expand this to a larger bayer pattern. Or (more likely), scale
// the threshold based on the target quantization levels.
static int Bayer4[4][4] = {
    {15, 7, 13, 5}, {3, 11, 1, 9}, {12, 4, 14, 6}, {0, 8, 2, 10}};

void Image::OrderedDither(int nbits) { /* WORK HERE  (Extra Credit) */ }

/* Error-diffusion parameters */
const double ALPHA = 7.0 / 16.0, BETA = 3.0 / 16.0, GAMMA = 5.0 / 16.0,
             DELTA = 1.0 / 16.0;

struct f_pix {
  double r;
  double g;
  double b;

  f_pix operator+(const f_pix &other) const {
    return f_pix{r + other.r, g + other.g, b + other.b};
  }

  f_pix operator*(double scalar) const {
    return f_pix{r * scalar, g * scalar, b * scalar};
  }
  void operator+=(const f_pix &other) {
    r += other.r;
    g += other.g;
    b += other.b;
  }
};

void propogate_error(int x, int y, f_pix err,
                     std::vector<std::vector<f_pix>> &image, bool going_right) {
  int width = image[0].size();
  int height = image.size();

  if (going_right) {
    // Standard Floyd-Steinberg pattern (left-to-right)
    if (x + 1 < width) {
      image[y][x + 1] += err * (7.0 / 16.0);
    }
    if (y + 1 < height) {
      if (x - 1 >= 0) {
        image[y + 1][x - 1] += err * (1.0 / 16.0);
      }
      image[y + 1][x] += err * (5.0 / 16.0);
      if (x + 1 < width) {
        image[y + 1][x + 1] += err * (3.0 / 16.0);
      }
    }
  } else {
    // Mirrored pattern (right-to-left)
    if (x - 1 >= 0) {
      image[y][x - 1] += err * (7.0 / 16.0);
    }
    if (y + 1 < height) {
      if (x + 1 < width) {
        image[y + 1][x + 1] += err * (1.0 / 16.0);
      }
      image[y + 1][x] += err * (5.0 / 16.0);
      if (x - 1 >= 0) {
        image[y + 1][x - 1] += err * (3.0 / 16.0);
      }
    }
  }
}

void Image::FloydSteinbergDither(int nbits) {
  std::vector<std::vector<f_pix>> f_image(Height(),
                                          std::vector<f_pix>(Width()));

  for (int x = 0; x < Width(); x++) {
    for (int y = 0; y < Height(); y++) {
      Pixel p = GetPixel(x, y);
      f_image[y][x] = f_pix{(double)p.r, (double)p.g, (double)p.b};
    }
  }

  this->export_depth = nbits;
  int maximum = (1 << nbits) - 1;
  double step = 255.0 / maximum;

  for (int y = 0; y < Height(); y++) {
    if (y % 2 == 0) {
      for (int x = 0; x < Width(); x++) {
        f_pix p = f_image[y][x];
        double level_r = round((p.r * maximum) / 255.0f);
        double level_g = round((p.g * maximum) / 255.0f);
        double level_b = round((p.b * maximum) / 255.0f);

        f_pix err = f_pix{
            p.r - level_r * step,
            p.g - level_g * step,
            p.b - level_b * step,
        };

        propogate_error(x, y, err, f_image, true);

        Pixel new_pixel = Pixel();
        new_pixel.SetClamp(level_r * step, level_g * step, level_b * step);
        // printf("pixel: %d %d %d : ", new_pixel.r, new_pixel.g, new_pixel.b);
        // printf("xy: %d %d rgb: %f %f %f \n", x, y, level_r, level_g,
        // level_b);
        this->SetPixel(x, y, new_pixel);
      }
    } else {
      for (int x = Width() - 1; x >= 0; x--) {
        f_pix p = f_image[y][x];
        double level_r = round((p.r * maximum) / 255.0f);
        double level_g = round((p.g * maximum) / 255.0f);
        double level_b = round((p.b * maximum) / 255.0f);

        f_pix err = f_pix{
            p.r - level_r * step,
            p.g - level_g * step,
            p.b - level_b * step,
        };

        propogate_error(x, y, err, f_image, false);

        Pixel new_pixel = Pixel();
        new_pixel.SetClamp(level_r * step, level_g * step, level_b * step);
        // printf("pixel: %d %d %d : ", new_pixel.r, new_pixel.g, new_pixel.b);
        // printf("xy: %d %d rgb: %f %f %f \n", x, y, level_r, level_g,
        // level_b);
        this->SetPixel(x, y, new_pixel);
      }
    }
  }
}

/* modifies the dst with the kernel*/
void Convolve(Image *src, Image *dst, std::vector<std::vector<double>> kernel,
              int edge_pattern) {
  int n = kernel.size() / 2;
  // Loop over
  for (int x = 0; x < src->Width(); x++) {
    for (int y = 0; y < src->Height(); y++) {
      double r = 0, g = 0, b = 0;

      // Loop over kernel
      for (int i = -n; i <= n; i++) {
        for (int j = -n; j <= n; j++) {
          int xx = std::min(std::max(x + i, 0), src->Width() - 1);
          int yy = std::min(std::max(y + j, 0), src->Height() - 1);

          Pixel p = src->GetPixel(xx, yy);
          double weight = kernel[i + n][j + n];
          r += weight * p.r;
          g += weight * p.g;
          b += weight * p.b;
        }
      }

      Pixel new_p = Pixel();
      new_p.SetClamp(r, g, b);
      dst->SetPixel(x, y, new_p);
    }
  }
}

// Gaussian blur with size nxn filter
void Image::Blur(int n) {
  Image *img_copy =
      new Image(*this); // This is will copying the image, so you can read the
                        // original values for filtering
  int size = 2 * n + 1;
  double sigma = n / 2.0;
  double sum = 0.0;
  std::vector<std::vector<double>> kernel(size, std::vector<double>(size));

  for (int i = -n; i <= n; i++) {
    for (int j = -n; j <= n; j++) {
      double exponent = -(i * i + j * j) / (2 * sigma * sigma);
      double value = exp(exponent);
      kernel[i + n][j + n] = value;
      sum += value;
    }
  }

  // Normalize
  double t = 0;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      kernel[i][j] /= sum;
      t += kernel[i][j];
    }
  }

  Convolve(img_copy, this, kernel, 0);
  delete img_copy;
}

void Image::Sharpen(int n) {
  Image *blurred_image = new Image(*this);
  blurred_image->Blur(2);
  for (int x = 0; x < this->Width(); x++) {
    for (int y = 0; y < this->Height(); y++) {
      Pixel orig_p = this->GetPixel(x, y);
      Pixel blur_p = blurred_image->GetPixel(x, y);
      double f = n / 10.0f;
      double r = orig_p.r + f * (orig_p.r - blur_p.r);
      double g = orig_p.g + f * (orig_p.g - blur_p.g);
      double b = orig_p.b + f * (orig_p.b - blur_p.b);

      Pixel new_p = Pixel();
      new_p.SetClamp(r, g, b);
      this->SetPixel(x, y, new_p);
    }
  }
  delete blurred_image;
}
// Image *img_copy = new Image(*this);
// int m = 1;
// int size = 3;
// std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
//
// for (int i = -m; i <= m; i++) {
//   for (int j = -m; j <= m; j++) {
//     if (i == 0 and j == 0) {
//       kernel[i + m][j + m] = 5;
//     } else if (abs(j) + abs(i) == 1) {
//       kernel[i + m][j + m] = -1;
//     } else {
//       kernel[i + m][j + m] = 0;
//     }
//   }
// }
//
// Convolve(img_copy, this, kernel, 0);
// delete img_copy;

void Image::EdgeDetect() {
  Image *img_copy = new Image(*this);
  int n = 1;
  int size = 3;
  std::vector<std::vector<double>> kernel(size, std::vector<double>(size));

  for (int i = -n; i <= n; i++) {
    for (int j = -n; j <= n; j++) {
      if (i == 0 and j == 0) {
        kernel[i + n][j + n] = 8;
      } else {
        kernel[i + n][j + n] = -1;
      }
    }
  }

  Convolve(img_copy, this, kernel, 0);
  delete img_copy;
}

Image *Image::Scale(double sx, double sy) {
  Image *img_copy = new Image(Width() * sx, Height() * sy);

  float src_cx = Width() / 2.0f;
  float src_cy = Height() / 2.0f;
  float dst_cx = img_copy->Width() / 2.0f;
  float dst_cy = img_copy->Height() / 2.0f;

  for (int x = 0; x < img_copy->Width(); x++) {
    for (int y = 0; y < img_copy->Height(); y++) {
      float dx = x - dst_cx;
      float dy = y - dst_cy;

      float u = dx / sx + src_cx;
      float v = dy / sy + src_cy;

      img_copy->GetPixel(x, y) = Sample(u, v);
    }
  }
  return img_copy;
}

Image *Image::Rotate(double angle) {
  Image *img_copy = new Image(*this);

  float cx = Width() / 2.0f;
  float cy = Height() / 2.0f;

  double cos_a = cos(angle);
  double sin_a = sin(angle);

  for (int x = 0; x < Width(); x++) {
    for (int y = 0; y < Height(); y++) {
      float dx = x - cx;
      float dy = y - cy;

      float u = dx * cos_a + dy * sin_a;
      float v = -dx * sin_a + dy * cos_a;

      u += cx;
      v += cy;

      // printf("u,v %f %f\n", u, v);

      img_copy->GetPixel(x, y) = Sample(u, v);
    }
  }

  return img_copy;
}

void Image::Fun() { /* WORK HERE */ }

/**
 * Image Sample
 **/
void Image::SetSamplingMethod(int method) {
  assert((method >= 0) && (method < IMAGE_N_SAMPLING_METHODS));
  sampling_method = method;
}

Pixel GaussianSample(int x, int y, Image &image) {
  int n = 2;
  int size = 2 * n + 1;
  double sigma = n / 2.0;
  double sum = 0.0;

  if (not image.ValidCoord(x, y)) {
    return Pixel();
  }

  std::vector<std::vector<double>> kernel(size, std::vector<double>(size));
  for (int i = -n; i <= n; i++) {
    for (int j = -n; j <= n; j++) {
      double exponent = -(i * i + j * j) / (2 * sigma * sigma);
      double value = exp(exponent);
      kernel[i + n][j + n] = value;
      sum += value;
    }
  }

  // Normalize
  double t = 0;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      kernel[i][j] /= sum;
      t += kernel[i][j];
    }
  }

  double r = 0, g = 0, b = 0;
  for (int i = -n; i <= n; i++) {
    for (int j = -n; j <= n; j++) {
      int xx = std::min(std::max(x + i, 0), image.Width() - 1);
      int yy = std::min(std::max(y + j, 0), image.Height() - 1);

      // will fail if not valid
      Pixel p;

      try {
        p = image.GetPixel(xx, yy);
      } catch (const std::out_of_range &e) {
        return Pixel();
      }

      double weight = kernel[i + n][j + n];
      r += weight * p.r;
      g += weight * p.g;
      b += weight * p.b;
    }
  }
  Pixel p = Pixel();
  p.SetClamp(r, g, b);
  return p;
}

Pixel Image::Sample(double u, double v) {
  if (sampling_method == IMAGE_SAMPLING_POINT) { // Nearest Neighbor
    // printf("samp %d %d\n", (int)u , (int)v );
    try {
      Pixel p = GetPixel((int)u, (int)v);
      // printf("found %d %d %d\n", p.r, p.g, p.b);
      return p;
    } catch (const std::out_of_range &e) {
      // printf("nope %d %d\n", (int)u , (int)v );
      return Pixel();
    }

  } else if (sampling_method == IMAGE_SAMPLING_BILINEAR) { // Bilinear
    // Get the integer and fractional parts
    int x0 = (int)floor(u);
    int y0 = (int)floor(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = u - x0; // Fractional part in x
    float fy = v - y0; // Fractional part in y
    //
    static int count;

    // Check bounds and get the 4 neighboring pixels
    if (x0 < 0 || x1 >= Width() || y0 < 0 || y1 >= Height()) {
      return Pixel(); // Out of bounds
    }

    Pixel p00 = GetPixel(x0, y0); // Top-left
    Pixel p10 = GetPixel(x1, y0); // Top-right
    Pixel p01 = GetPixel(x0, y1); // Bottom-left
    Pixel p11 = GetPixel(x1, y1); // Bottom-right

    // Bilinear interpolation formula
    double r = (1 - fx) * (1 - fy) * p00.r + fx * (1 - fy) * p10.r +
               (1 - fx) * fy * p01.r + fx * fy * p11.r;

    double g = (1 - fx) * (1 - fy) * p00.g + fx * (1 - fy) * p10.g +
               (1 - fx) * fy * p01.g + fx * fy * p11.g;

    double b = (1 - fx) * (1 - fy) * p00.b + fx * (1 - fy) * p10.b +
               (1 - fx) * fy * p01.b + fx * fy * p11.b;

    Pixel result = Pixel();
    result.SetClamp(r, g, b);
    return result;
  } else if (sampling_method == IMAGE_SAMPLING_GAUSSIAN) { // Gaussian
    // return the gaussian-weighted average
    return GaussianSample(u, v, *this);
  }
  return Pixel(); // we should never be here
}
