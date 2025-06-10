#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <string>

int main() {
    try {
        // Load the ONNX model (correct path from build directory)
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sentence_transformer");
        Ort::SessionOptions session_options;
        Ort::Session session(env, "../../onnx-model/model.onnx", session_options);

        // Prepare input data
        std::vector<int64_t> input_ids = {101, 2054, 2003, 1996, 2313, 102}; // tokenized text
        std::vector<int64_t> attention_mask = {1, 1, 1, 1, 1, 1};
        std::vector<int64_t> input_shape{1, static_cast<int64_t>(input_ids.size())};

        // Input names
        const char* input_names[] = {"input_ids", "attention_mask"};
        
        // Try the actual output names we found
        const char* output_names_options[][1] = {
            {"sentence_embedding"},     // This is what we want for sentence embeddings
            {"token_embeddings"},       // Individual token embeddings
        };

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Try each possible output name
        for (int i = 0; i < 2; ++i) {
            try {
                std::cout << "Trying output name: " << output_names_options[i][0] << std::endl;
                
                // Create input tensors (need to recreate for each attempt)
                auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info, input_ids.data(), input_ids.size(), 
                    input_shape.data(), input_shape.size());
                
                auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info, attention_mask.data(), attention_mask.size(),
                    input_shape.data(), input_shape.size());

                std::vector<Ort::Value> input_tensors;
                input_tensors.push_back(std::move(input_ids_tensor));
                input_tensors.push_back(std::move(attention_mask_tensor));

                // Run inference
                auto outputs = session.Run(Ort::RunOptions{nullptr}, 
                                         input_names, input_tensors.data(), input_tensors.size(),
                                         output_names_options[i], 1);

                // Extract embeddings from outputs
                float* output_data = outputs[0].GetTensorMutableData<float>();
                auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                
                std::cout << "SUCCESS! Output shape: ";
                for (size_t j = 0; j < output_shape.size(); ++j) {
                    std::cout << output_shape[j] << " ";
                }
                std::cout << std::endl;

                // Print first few embedding values
                std::cout << "First 10 embedding values: ";
                int values_to_print = std::min(10, static_cast<int>(output_shape.back()));
                for (int j = 0; j < values_to_print; ++j) {
                    std::cout << output_data[j] << " ";
                }
                std::cout << std::endl;
                
                // If this is sentence_embedding, we're done!
                if (std::string(output_names_options[i][0]) == "sentence_embedding") {
                    std::cout << "\nFound sentence embeddings! Embedding dimension: " << output_shape.back() << std::endl;
                    return 0;
                }
                
            } catch (const Ort::Exception& e) {
                std::cout << "Failed with output name '" << output_names_options[i][0] 
                         << "': " << e.what() << std::endl;
                continue;
            }
        }

        std::cout << "All output names failed!" << std::endl;
        return -1;

    } catch (const Ort::Exception& e) {
        std::cout << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}