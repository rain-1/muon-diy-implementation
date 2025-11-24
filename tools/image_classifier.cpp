#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 32 * 32 * 3;

struct ImageStats {
    float r_mean;
    float g_mean;
    float b_mean;
    float brightness;
    float saturation;
};

ImageStats compute_stats(const std::vector<float>& data) {
    if (data.size() != static_cast<size_t>(kInputSize)) {
        throw std::runtime_error("Expected " + std::to_string(kInputSize) + " floats");
    }

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float saturation_accum = 0.0f;
    float brightness_accum = 0.0f;

    for (int i = 0; i < 32 * 32; ++i) {
        float r = data[i];
        float g = data[i + 32 * 32];
        float b = data[i + 2 * 32 * 32];

        r_sum += r;
        g_sum += g;
        b_sum += b;

        float max_c = std::max({r, g, b});
        float min_c = std::min({r, g, b});
        saturation_accum += (max_c == 0.0f) ? 0.0f : (max_c - min_c) / max_c;
        brightness_accum += (r + g + b) / 3.0f;
    }

    const float inv_pixels = 1.0f / static_cast<float>(32 * 32);
    return {r_sum * inv_pixels,
            g_sum * inv_pixels,
            b_sum * inv_pixels,
            brightness_accum * inv_pixels,
            saturation_accum * inv_pixels};
}

struct LabelScore {
    std::string label;
    float score;
};

std::vector<LabelScore> predict(const std::vector<float>& input) {
    static const std::array<std::string, 6> labels = {
        "forest",
        "ocean",
        "sunset",
        "snow",
        "indoor",
        "city"
    };

    // Simple handcrafted weights over five features (r_mean, g_mean, b_mean, brightness, saturation)
    static const std::array<std::array<float, 5>, 6> weights = {
        std::array<float, 5>{-0.1f, 0.8f, -0.2f, 0.2f, 0.6f},  // forest
        std::array<float, 5>{-0.2f, 0.1f, 1.0f, 0.3f, 0.5f},   // ocean
        std::array<float, 5>{0.9f, 0.2f, -0.1f, 0.4f, 0.3f},  // sunset
        std::array<float, 5>{0.4f, 0.4f, 0.5f, 0.9f, -0.2f},  // snow
        std::array<float, 5>{0.3f, 0.3f, 0.1f, 0.2f, 0.1f},   // indoor
        std::array<float, 5>{0.2f, 0.3f, 0.4f, 0.5f, 0.2f}    // city
    };

    static const std::array<float, 6> bias = {0.05f, -0.1f, 0.1f, 0.0f, -0.05f, 0.02f};

    ImageStats stats = compute_stats(input);
    const std::array<float, 5> features = {stats.r_mean, stats.g_mean, stats.b_mean, stats.brightness, stats.saturation};

    std::vector<LabelScore> scores;
    scores.reserve(labels.size());

    for (size_t i = 0; i < labels.size(); ++i) {
        float score = bias[i];
        for (size_t f = 0; f < features.size(); ++f) {
            score += weights[i][f] * features[f];
        }
        scores.push_back({labels[i], score});
    }

    std::partial_sort(scores.begin(), scores.begin() + 3, scores.end(), [](const LabelScore& a, const LabelScore& b) {
        return a.score > b.score;
    });

    scores.resize(3);
    return scores;
}

std::vector<float> load_input(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open input file: " + path);
    }
    std::vector<float> data(kInputSize);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (static_cast<size_t>(file.gcount()) != data.size() * sizeof(float)) {
        throw std::runtime_error("Input file did not contain " + std::to_string(kInputSize) + " floats");
    }
    return data;
}

void print_json(const std::vector<LabelScore>& scores) {
    std::cout << "{\n  \"labels\": [\n";
    for (size_t i = 0; i < scores.size(); ++i) {
        std::cout << "    {\"label\": \"" << scores[i].label << "\", \"score\": " << scores[i].score << "}";
        if (i + 1 != scores.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <float32_image_path>" << std::endl;
        return 1;
    }

    try {
        auto input = load_input(argv[1]);
        auto scores = predict(input);
        print_json(scores);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
