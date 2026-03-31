//
// ResNet 18 with group instead of batch norm.
//

#ifndef PPO_CPP_CUSTOM_RESNET_H
#define PPO_CPP_CUSTOM_RESNET_H

#include <torch/torch.h>
using namespace torch;

struct BasicBlockImpl : nn::Module {
    int expansion = 1;

    nn::Conv2d conv1{nullptr};
    nn::GroupNorm gn1{nullptr};
    nn::Conv2d conv2{nullptr};
    nn::GroupNorm gn2{nullptr};
    nn::Sequential shortcut{nullptr};

    BasicBlockImpl(const int in_planes, const int planes, const int stride = 1) {

        // Use a divisor that works for all channel sizes, e.g. min(32, planes)
        const int num_groups = std::min(32, planes);
        // Or assert divisibility
        assert(planes % 32 == 0);
        // Conv1
        conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)));
        gn1 = register_module("gn1", nn::GroupNorm(nn::GroupNormOptions(num_groups, planes)));

        // Conv2
        conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)));
        gn2 = register_module("gn2", nn::GroupNorm(nn::GroupNormOptions(num_groups, planes)));

        // Shortcut
        shortcut = register_module("shortcut", nn::Sequential());
        if (stride != 1 || in_planes != expansion * planes) {
            shortcut->push_back(nn::Conv2d(nn::Conv2dOptions(in_planes, expansion * planes, 1).stride(stride).bias(false)));
            shortcut->push_back(nn::GroupNorm(nn::GroupNormOptions(num_groups, expansion * planes)));
        }
    }

    Tensor forward(const Tensor& x) {
        Tensor out = relu(gn1->forward(conv1->forward(x)));
        out = gn2->forward(conv2->forward(out));

        // Emulate Python's empty nn.Sequential behavior
        if (!shortcut->is_empty()) {
            out += shortcut->forward(x);
        } else {
            out += x;
        }
        out = relu(out);

        return out;
    }
};
TORCH_MODULE(BasicBlock);


struct ResNetRLImpl : nn::Module {
    int in_planes = 64;

    nn::Conv2d conv1{nullptr};
    nn::GroupNorm gn1{nullptr};
    nn::ReLU relu{nullptr};
    nn::MaxPool2d maxpool{nullptr};

    nn::Sequential layer1{nullptr};
    nn::Sequential layer2{nullptr};
    nn::Sequential layer3{nullptr};
    nn::Sequential layer4{nullptr};
    nn::Sequential layer5{nullptr};

    ResNetRLImpl(const int in_channel, const std::vector<int>& num_blocks) {

        // Standard Stem
        conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(in_channel, 64, 7).stride(2).padding(3).bias(false)));
        gn1 = register_module("gn1", nn::GroupNorm(nn::GroupNormOptions(32, 64)));
        relu = register_module("relu", nn::ReLU(nn::ReLUOptions().inplace(true)));
        maxpool = register_module("maxpool", nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // Residual Layers
        layer1 = register_module("layer1", _make_layer(64, num_blocks[0], 1));
        layer2 = register_module("layer2", _make_layer(128, num_blocks[1], 2));
        layer3 = register_module("layer3", _make_layer(256, num_blocks[2], 2));
        layer4 = register_module("layer4", _make_layer(512, num_blocks[3], 2));
        layer5 = register_module("layer5", _make_layer(512, num_blocks[4], 2));

        _initialize_weights();
    }

    nn::Sequential _make_layer(const int planes, const int num_blocks, const int stride) {
        std::vector<int> strides(num_blocks, 1);
        strides[0] = stride;

        torch::nn::Sequential layers;
        for (int s : strides) {
            layers->push_back(BasicBlock(in_planes, planes, s));
            in_planes = planes * 1; // 1 is BasicBlock::expansion
        }
        return layers;
    }

    void _initialize_weights() const {
        // Iterate over all modules recursively
        for (const auto& module : modules(/*include_self=*/false)) {
            auto conv = module->as<nn::Conv2dImpl>();
            // Check if the module is a Conv2d
            if (conv != nullptr) {
                nn::init::orthogonal_(conv->weight, nn::init::calculate_gain(kReLU));
                if (conv->bias.defined()) {
                    nn::init::constant_(conv->bias, 0.0f);
                }
            }
            // Check if the module is a GroupNorm
            else {
                auto gn = module->as<nn::GroupNormImpl>();
                if (gn != nullptr) {
                    if (gn->weight.defined()) {
                        nn::init::constant_(gn->weight, 1.0f);
                    }
                    if (gn->bias.defined()) {
                        nn::init::constant_(gn->bias, 0.0f);
                    }
                }
            }
        }
    }

    Tensor forward(const Tensor& x) {
        Tensor out = conv1->forward(x);
        out = gn1->forward(out);
        out = relu->forward(out);
        out = maxpool->forward(out);

        out = layer1->forward(out);
        out = layer2->forward(out);
        out = layer3->forward(out);
        out = layer4->forward(out);
        out = layer5->forward(out);

        return out;
    }
};
TORCH_MODULE(ResNetRL);


// ==========================================
// Helper Function
// ==========================================

// Returns a ResNet18 model optimized for RL with GroupNorm.
inline ResNetRL ResNet22_PPO(int in_channel) {
    return ResNetRL(in_channel, vector<int>{2, 2, 2, 2, 2});
}

#endif //PPO_CPP_CUSTOM_RESNET_H