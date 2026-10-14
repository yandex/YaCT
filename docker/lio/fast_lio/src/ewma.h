// Added by YANDEX LLC on 2026-06-08 for local pose smoothing.

#pragma once

#include <cassert>
#include <fast_lio/Pose6D.h>
#include <tf/LinearMath/Quaternion.h>


template<typename T>
class EWMAFilter {
public:
    virtual void Reset() = 0;
    virtual T Step(const T& sample) = 0;
    virtual ~EWMAFilter() = default;
};


class PosFilter : public EWMAFilter<fast_lio::Pose6D::_pos_type> {
public:
    using Pos = fast_lio::Pose6D::_pos_type;
    explicit PosFilter(double alpha) : alpha_{alpha} {
        assert(alpha > 0. && alpha <= 1.);
        Reset();
    }
    ~PosFilter() override = default;

    void Reset() override {
        inited_ = false;
    }

    Pos Step(const Pos& sample) override {
        if (!inited_) {
            accumulated_ = sample;
            inited_ = true;
            return sample;
        }

        Pos result{};
        for (int i = 0; i < result.size(); ++i) {
            result[i] = sample[i] * alpha_ + accumulated_[i] * (1. - alpha_);
        }
        accumulated_ = result;
        return result;
    }

private:
    Pos accumulated_;
    const double alpha_;
    bool inited_;
};


class QuatFilter : public EWMAFilter<tf::Quaternion> {
public:
    explicit QuatFilter(double alpha) : alpha_{alpha} {
        assert(alpha > 0. && alpha <= 1.);
        Reset();
    }
    ~QuatFilter() override = default;

    void Reset() override {
        inited_ = false;
    }

    tf::Quaternion Step(const tf::Quaternion& sample) override {
        if (!inited_) {
            accumulated_ = sample;
            inited_ = true;
            return sample;
        }

        tf::Quaternion result = accumulated_.slerp(sample, alpha_);
        accumulated_ = result;
        return result;
    }

private:
    tf::Quaternion accumulated_;
    const double alpha_;
    bool inited_;
};
