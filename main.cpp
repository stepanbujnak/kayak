#include <iostream>
#include <vector>
#include <cassert>

#include <opencv2/opencv.hpp>

// Sectors is our constant determining how each character will be split into a
// grid. Here we've got 3x3 grid, each cell is going to be an individual input
// to the ANN.
static const cv::Size Sectors(3, 3);

// Structure capturing each character.
struct Char {
  
  int x; // X coordinate of the top left corner of the character in the
         // original image
  
  int y; // Y coordinate of the top left corner of the character in the
         // original image
  int w; // Character width
  int h; // Character height
  int predicted; // The ASCII value of the character predicted by the ANN
  cv::Mat mat; // Image of the character
};

// Helper structure used by query search
struct Location {
  int x; // Column of the text matrix
  int y; // Row of the text matrix
  int dir_x; // Search direction for `x`
  int dir_y; // Search direction for `y`

  Location(int x, int y, int dir_x, int dir_y)
      : x(x), y(y), dir_x(dir_x), dir_y(dir_y) {}
};

// A matrix with each character indexed by its actual position.
// This is necessary for the search that will be conducted after character
// recognition.
typedef std::vector<std::vector<Char>> Text;

//----------------------------------------------------------------------------
// isColBlank
//----------------------------------------------------------------------------
// Iterates over all columns in a rowspecified by the `row_idx` parameter and
// returns true if the row is blank, i.e. all the pixels in the row are white.
static bool
isRowBlank(const cv::Mat &mat, int row_idx) {
  const auto &row = mat.ptr<uint8_t>(row_idx);

  for (int i = 0; i < mat.cols; ++i) {
    if (row[i] == 0) {
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
// isColBlank
//----------------------------------------------------------------------------
// Iterates over all rows in a column specified by the `col_idx` parameter and
// returns true if the column is blank, i.e. all the pixels in the column
// are white.
static bool
isColBlank(const cv::Mat &mat, int col_idx) {
  for (int i = 0; i < mat.rows; ++i) {
    const auto &row = mat.ptr<uint8_t>(i);

    if (row[col_idx] == 0) {
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
// calculateInputs
//----------------------------------------------------------------------------
// The function resizes the character image into size specified by `Sectors`
// e.g. if sectors = 3x3 then the image will have 9 pixels.
// Each pixel is then used as an input to a neural network.
static void
calculateInputs(const cv::Mat &mat, cv::Mat &inputs) {
  cv::Mat small, flat;

  cv::resize(mat, small, Sectors);
  flat = small.reshape(0, 1);

  assert(flat.cols == inputs.cols);

  for (int i = 0; i < flat.cols; ++i) {
    inputs.at<double>(0, i) = static_cast<double>(flat.at<uint8_t>(0, i));
  }
}

//----------------------------------------------------------------------------
// detectText
//----------------------------------------------------------------------------
// Here we merely extract character image from the large, global image. Since
// the image is thresholded we can separate characters by empty (white) rows
// and columns.
static void
detectText(const cv::Mat &mat, Text &text) {
  std::vector<std::pair<int, cv::Mat>> lines;
  bool was_blank = true;
  int line_start = 0;
  int char_start = 0;

  // Detect lines
  for (int i = 0; i < mat.rows; ++i) {
    bool is_blank = isRowBlank(mat, i);

    if (!is_blank && was_blank) {
      line_start = i;
      was_blank = false;
    } else if (is_blank && !was_blank) {
      auto line_mat = mat(cv::Rect(0, line_start, mat.cols, i - line_start));
      lines.push_back(std::make_pair(line_start, line_mat));
      was_blank = true;
    }
  }

  was_blank = true;

  // Detect chars
  for (const auto &line: lines) {
    std::vector<Char> chars;

    for (int i = 0; i < line.second.cols; ++i) {
      bool is_blank = isColBlank(line.second, i);

      if (!is_blank && was_blank) {
        char_start = i;
        was_blank = false;
      } else if (is_blank && !was_blank) {
        Char cr;

        // Save locations
        cr.x = char_start;
        cr.y = line.first;
        cr.w = i - char_start;
        cr.h = line.second.rows;
        cr.mat = line.second(cv::Rect(cr.x, 0, cr.w, cr.h));

        chars.push_back(cr);
        was_blank = true;
      }
    }

    text.push_back(chars);
  }
}

//----------------------------------------------------------------------------
// readText
//----------------------------------------------------------------------------
// Initialize and train the ANN
// Our topology is going to be:
//   Inputs: the area of sectors. If sectors is 3x3 then inputs is 9
//   Hidden: Rule of thumb: 1 layer, (size(inputs) + size(outputs)) / 2
//   Outputs: 1 (we only need one character)
//---
// After the ANN is initialized and train we predict by running the extracted
// characters through the neural network and then assigning label to the nearest
// match
static void
readText(Text &text, const cv::Mat &train_in, const cv::Mat &train_out) {
  auto nb_inputs = Sectors.area();
  CvANN_MLP ann((cv::Mat_<int>(1, 3) << nb_inputs, ((nb_inputs + 1) / 2), 1));
  ann.train(train_in, train_out, cv::Mat());

  // Predict
  for (int i = 0; i < text.size(); ++i) {
    for (int j = 0; j < text[i].size(); ++j) {
      cv::Mat inputs(1, Sectors.area(), CV_64F);
      calculateInputs(text[i][j].mat, inputs);

      cv::Mat outputs(1, 1, CV_64F);
      ann.predict(inputs, outputs);

      auto predicted = outputs.at<double>(0, 0);

      int nearest_dst = 9999;
      int nearest_val = -1;
      for (int k = 0; k < train_out.rows; ++k) {
        int dst = std::abs(predicted - train_out.at<double>(k, 0));

        if (dst < nearest_dst) {
          nearest_val = train_out.at<double>(k, 0);
          nearest_dst = dst;
        }
      }

      text[i][j].predicted = nearest_val;
    }
  }
}

//----------------------------------------------------------------------------
// searchInDir
//----------------------------------------------------------------------------
// Searches the character matrix in direction specified by the `loc` parameter
// for specified query.
// It also does some boundary checking to be sure that we don't run out of
// canvas.
static bool
searchInDir(const Text &text, const std::string &query, const Location &loc) {
  int x = loc.x + loc.dir_x;
  int y = loc.y + loc.dir_y;

  int last_x = loc.x + (query.size() * loc.dir_x);
  int last_y = loc.y + (query.size() * loc.dir_y);

  if (last_y < 0 || last_y >= text.size() ||
      last_x < 0 || last_x >= text[last_y].size()) {
    return false;
  }

  for (int i = 1; i < query.size(); ++i) {
    if (query[i] != text[x][y].predicted) {
      return false;
    }

    x += loc.dir_x;
    y += loc.dir_y;
  }

  return true;
}

//----------------------------------------------------------------------------
// colorMatch
//----------------------------------------------------------------------------
// Create red circle around the character. We can do this because we stored
// character position in the actual image before.
static void
colorMatch(cv::Mat &image, const Text &text, const Location &loc, int len) {
  int x = loc.x;
  int y = loc.y;

  for (int i = 0; i < len; ++i) {
    auto cr = text[x][y];
    auto center_x = cr.x + (cr.w / 2);
    auto center_y = cr.y + (cr.h / 2);

    cv::Point center(center_x, center_y);
    cv::circle(image, center, cr.h, cv::Scalar(0, 0, 255), 2);

    x += loc.dir_x;
    y += loc.dir_y;
  }
}

//----------------------------------------------------------------------------
// colorMatches
//----------------------------------------------------------------------------
// Run search and then circle all matched characters
static void
colorMatches(cv::Mat &image, const Text &text, const std::string &query) {
  static const std::vector<std::pair<int, int>> directions {
    {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}
  };

  for (int i = 0; i < text.size(); ++i) {
    for (int j = 0; j < text[i].size(); ++j) {
      // Only if the first letter matches
      if (text[i][j].predicted == query[0]) {
        for (const auto &dir: directions) {
          Location location(i, j, dir.first, dir.second);

          if (searchInDir(text, query, location)) {
            colorMatch(image, text, location, query.size());
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
// Main
//----------------------------------------------------------------------------
int
main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <image-path>" << std::endl;
    return -1;
  }

  // Load original image, in colors. We need RGB image so we can draw
  // red circles later on.
  cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (!image.data) {
    std::cerr << "No image data" << std::endl;
    return -1;
  }

  // Convert colored image into grayscale
  cv::Mat grayscale;
  cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

  // Set pixels to either black or white, by a threshold
  cv::Mat binary;
  cv::threshold(grayscale, binary, 127, 255, cv::THRESH_BINARY);

  // Recognize all the characters
  Text text;
  detectText(binary, text);

  // Handpicked characters used for training of the ANN.
  auto K = text[0][3].mat;
  auto A = text[0][0].mat;
  auto Y = text[0][2].mat;

  // Load inputs to corresponding data structures.
  cv::Mat train_inputs(3, Sectors.area(), CV_64F);
  cv::Mat row;

  row = train_inputs.row(0);
  calculateInputs(K, row);
  row = train_inputs.row(1);
  calculateInputs(A, row);
  row = train_inputs.row(2);
  calculateInputs(Y, row);

  // Load outputs to corresponding data structures.
  cv::Mat train_outputs(3, 1, CV_64F);
  train_outputs.at<double>(0, 0) = 'K';
  train_outputs.at<double>(1, 0) = 'A';
  train_outputs.at<double>(2, 0) = 'Y';

  // Do the actual prediction
  readText(text, train_inputs, train_outputs);

  // Color all connected characters matching the query string
  colorMatches(image, text, "KAYAK");

  // Display colored image
  /*cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);*/
  cv::imwrite("matched.jpg", image);

  return 0;
}
