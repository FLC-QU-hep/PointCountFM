#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <H5Cpp.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace std::chrono;

#define duration(start)                                                        \
    duration_cast<microseconds>(high_resolution_clock::now() - start).count()

void save_dict(const std::string &file_name,
               const std::unordered_map<std::string, at::Tensor> &dict,
               std::string top_group = "") {
    H5::H5File file(file_name, H5F_ACC_TRUNC);
    if (!top_group.empty()) {
        file.createGroup(top_group);
    }

    for (const auto &pair : dict) {
        const at::Tensor &tensor = pair.second;
        std::string dataset_name;
        if (top_group.empty()) {
            dataset_name = pair.first;
        } else {
            dataset_name = top_group + "/" + pair.first;
        }

        c10::IntArrayRef shape = tensor.sizes();
        int num_dims = (int)(tensor.dim());
        c10::ScalarType dtype = tensor.scalar_type();
        H5::PredType memCompType(H5::PredType::NATIVE_CHAR);
        switch (dtype) {
        case at::ScalarType::Float:
            memCompType = H5::PredType::NATIVE_FLOAT;
            break;
        case at::ScalarType::Double:
            memCompType = H5::PredType::NATIVE_DOUBLE;
            break;
        case at::ScalarType::Int:
            memCompType = H5::PredType::NATIVE_INT;
            break;
        case at::ScalarType::Long:
            memCompType = H5::PredType::NATIVE_LONG;
            break;
        case at::ScalarType::Short:
            memCompType = H5::PredType::NATIVE_SHORT;
            break;
        case at::ScalarType::Char:
            memCompType = H5::PredType::NATIVE_CHAR;
            break;
        case at::ScalarType::Byte:
            memCompType = H5::PredType::NATIVE_UCHAR;
            break;
        default:
            throw std::runtime_error("Unsupported data type");
        }

        H5::DataSpace data_space(num_dims, (hsize_t *)(shape.data()));
        H5::DataSet dataset =
            file.createDataSet(dataset_name, memCompType, data_space);

        dataset.write(tensor.data_ptr(), memCompType, data_space);
    }

    file.close();
}

inline torch::Tensor sample_random_energy(int num_sample, float e_min = 10.,
                                          float e_max = 90.) {
    return torch::rand({num_sample, 1}) * (e_max - e_min) + e_min;
}

int main(int argc, const char *argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: generate <path-to-exported-script-module>"
                  << std::endl;
        return -1;
    }
    c10::InferenceMode guard(true);
    auto start = high_resolution_clock::now();
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
        module.eval();
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model" << std::endl;
        return -1;
    }
    std::cout << std::setprecision(4);
    std::cout << "loading time: " << duration(start) / 1000.0 << "ms"
              << std::endl;

    std::vector<torch::jit::IValue> inputs;
    float input[4] = {50.};
    at::Tensor input_tensor = torch::from_blob(input, {1, 1});
    inputs.push_back(input_tensor);

    start = high_resolution_clock::now();
    for (int i = 0; i < 2; i++) {
        module.forward(inputs);
    }
    std::cout << "warm up time: " << duration(start) / 1000.0 << "ms"
              << std::endl;

    torch::set_num_threads(1);

    int N = 100;
    if (argc == 3) {
        N = std::stoi(argv[2]);
    }
    torch::jit::Stack stack;
    at::TensorList tensors;

    at::Tensor e = sample_random_energy(N);
    std::cout << "e shape: " << e.sizes()[0] << " x " << e.sizes()[1]
              << std::endl;
    std::vector<at::Tensor> num_points_vector;
    start = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        inputs.clear();
        inputs.push_back(torch::concat({e[i]}).unsqueeze(0));
        at::Tensor num_points = module.forward(inputs).toTensor();
        num_points_vector.push_back(num_points);
        if ((i + 1) % 10000 == 0) {
            std::cout << std::endl;
            std::cout << "processed " << (i + 1) << "/" << N << std::endl;
            std::cout << "total time: " << duration(start) / 1000.0 << "ms"
                      << std::endl;
            std::cout << "average time: " << duration(start) / 1000.0 / (i + 1)
                      << "ms" << std::endl;
        }
    }
    if (N % 10000 != 0) {
        std::cout << "processed " << N << "/" << N << std::endl;
        std::cout << "total time: " << duration(start) / 1000.0 << "ms"
                  << std::endl;
        std::cout << "average time: " << duration(start) / 1000.0 / N << "ms"
                  << std::endl;
    }

    start = high_resolution_clock::now();
    std::unordered_map<std::string, at::Tensor> dict;
    std::string file_name = "output.h5";
    dict["energy"] = e;
    dict["num_points"] = torch::concat(num_points_vector);
    save_dict(file_name, dict);
    std::cout << std::endl;
    std::cout << "saving time: " << duration(start) / 1000.0 << "ms"
              << std::endl;

    return 0;
}
