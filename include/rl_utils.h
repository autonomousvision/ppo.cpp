#ifndef RL_UTILS_H
#define RL_UTILS_H

#include <torch/torch.h>
#include <cmath>
#include <numbers>
#include <ATen/native/Distributions.h>
#include <ATen/ops/_sample_dirichlet.h>
using namespace std;

class Distribution {
public:
    [[nodiscard]] virtual torch::Tensor log_prob(const torch::Tensor &value) const = 0;
    [[nodiscard]] virtual torch::Tensor sample(const std::optional<at::Generator>& generator) const = 0;
    [[nodiscard]] virtual torch::Tensor entropy() const = 0;

    virtual ~Distribution() = default;
};

const double lz = log(sqrt(2.0 * numbers::pi));
class Normal final : public Distribution {
    torch::Tensor mean_, stddev_, var_, log_std_;
public:
    Normal(const torch::Tensor &mean, const torch::Tensor &std) :
    mean_(mean), stddev_(std), var_(std * std), log_std_(std.log())
    {}


    [[nodiscard]] torch::Tensor rsample() const {
        const auto eps = torch::randn(1, mean_.device());
        return mean_ + eps * stddev_;
    }

    [[nodiscard]] torch::Tensor sample(const std::optional<at::Generator>& generator) const override {
        torch::NoGradGuard no_grad;
        return at::normal(mean_, stddev_, generator);
    }

    [[nodiscard]] torch::Tensor log_prob(const torch::Tensor &value) const override {
        return -((value - mean_) * (value - mean_)) / (2 * var_) - log_std_ - lz;
    }

    [[nodiscard]] torch::Tensor entropy() const override {
        return 0.5 + 0.5 * log(2 * numbers::pi) + torch::log(stddev_);
    }
};

class Dirichlet final : public Distribution {
    torch::Tensor concentration_;
public:
    Dirichlet() : concentration_(torch::zeros({1})) {
    }

    explicit Dirichlet(const torch::Tensor &concentration) :
    concentration_(concentration) {
    }


    [[nodiscard]] torch::Tensor sample(const std::optional<at::Generator>& generator) const override {
        torch::NoGradGuard no_grad;
        return at::_sample_dirichlet(concentration_, generator);
    }

    [[nodiscard]] torch::Tensor log_prob(const torch::Tensor &value) const override {
        torch::Tensor x = torch::special::xlogy(concentration_ - 1.0f, value).sum(-1);
        torch::Tensor y = torch::lgamma(concentration_.sum(-1));
        torch::Tensor z = torch::lgamma(concentration_).sum(-1);

        return torch::special::xlogy(concentration_ - 1.0f, value).sum(-1)
               + torch::lgamma(concentration_.sum(-1)) - torch::lgamma(concentration_).sum(-1);
    }

    [[nodiscard]] torch::Tensor entropy() const override {
        const auto k = concentration_.size(-1);
        const auto a0 = concentration_.sum(-1);
        return torch::lgamma(concentration_).sum(-1)
               - torch::lgamma(a0)
               - (k - a0) * torch::digamma(a0)
               - ((concentration_ - 1.0f) * torch::digamma(concentration_)).sum(-1);
    }
};

class Beta final : public Distribution {
    torch::Tensor alpha_, beta_;
    Dirichlet dirichlet_;
public:
    Beta(const torch::Tensor &alpha, const torch::Tensor &beta) :
    alpha_(alpha), beta_(beta) {
        const torch::Tensor alpha_beta = torch::stack({alpha, beta}, -1);
        dirichlet_ = Dirichlet(alpha_beta);
    }


    [[nodiscard]] torch::Tensor sample(const std::optional<at::Generator>& generator) const override {
        torch::NoGradGuard no_grad;
        return dirichlet_.sample(generator).select(-1, 0);
    }

    [[nodiscard]] torch::Tensor log_prob(const torch::Tensor &value) const override {
        const torch::Tensor heads_tails = torch::stack({value, 1.0f - value}, -1);
        return dirichlet_.log_prob(heads_tails);
    }

    [[nodiscard]] torch::Tensor entropy() const override {
        return dirichlet_.entropy();
    }

    [[nodiscard]] torch::Tensor mean() const {
        return alpha_ / (alpha_ + beta_);
    }

    // Determinsitc method proposed by https://arxiv.org/abs/2108.08265
    // Uses the mode in most cases, except for a < 1, beta < 1, where it uses constant values or the mean.
    [[nodiscard]] torch::Tensor roach_deterministic() const {
        torch::Tensor x = torch::zeros_like(alpha_, alpha_.device());
        x.index_put_({torch::indexing::Slice(), 1}, x.index({torch::indexing::Slice(), 1}) + 0.5f);

        torch::Tensor mask1 = (alpha_ > 1.0f) & (beta_ > 1.0f);
        x.index_put_({mask1}, (alpha_.index({mask1}) - 1.0f) / (alpha_.index({mask1}) + beta_.index({mask1}) - 2.0f));

        torch::Tensor mask2 = (alpha_ <= 1.0f) & (beta_ > 1.0f);
        x.index_put_({mask2}, 0.0f);

        torch::Tensor mask3 = (alpha_ > 1.0f) & (beta_ <= 1.0f);
        x.index_put_({mask3}, 1.0f);

        torch::Tensor mask4 = (alpha_ <= 1.0f) & (beta_ <= 1.0f);
        x.index_put_({mask4}, this->mean().index({mask4}));

        return x;
    }
};

#endif //RL_UTILS_H
