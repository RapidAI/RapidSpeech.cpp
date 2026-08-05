// rs-aoq-sim.cpp - AOQ-like transport policy simulator.
//
// This is not Alibaba AOQ wire compatibility. It models the public design
// shape: one session carrying realtime media and reliable data together, with
// realtime deadlines, redundancy, reliable retransmission, and bandwidth
// scheduling under packet loss.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct SimArgs {
    int duration_ms = 10000;
    int tick_ms = 10;
    int rtt_ms = 80;
    int jitter_ms = 12;
    int bandwidth_kbps = 160;
    int audio_bps = 32000;
    int audio_frame_ms = 20;
    int realtime_ttl_ms = 180;
    int data_kb = 256;
    int reliable_chunk_bytes = 1000;
    int realtime_copies = 2;
    double loss = 0.30;
    unsigned seed = 42;
};

struct RealtimeFrame {
    uint64_t seq = 0;
    int generated_ms = 0;
    int size_bytes = 0;
    bool delivered = false;
    bool expired = false;
    bool sent = false;
    int delivered_ms = -1;
};

struct ReliableChunk {
    uint64_t seq = 0;
    int size_bytes = 0;
    bool delivered = false;
    int next_send_ms = 0;
    int delivered_ms = -1;
    int sends = 0;
};

struct DeliveryEvent {
    enum Kind {
        Realtime,
        Reliable
    };

    Kind kind = Realtime;
    uint64_t seq = 0;
    int at_ms = 0;
};

struct Stats {
    uint64_t realtime_packets = 0;
    uint64_t realtime_packet_loss = 0;
    uint64_t reliable_packets = 0;
    uint64_t reliable_packet_loss = 0;
    uint64_t reliable_retransmits = 0;
    uint64_t realtime_bytes = 0;
    uint64_t reliable_bytes = 0;
};

static bool parse_int(const char *s, int &out)
{
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

static bool parse_double(const char *s, double &out)
{
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    out = v;
    return true;
}

static void print_usage(const char *prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --duration-ms <n>       Simulated call duration (default 10000)\n"
        << "  --bandwidth-kbps <n>    Link budget (default 160)\n"
        << "  --loss <0..1>           Packet loss probability (default 0.30)\n"
        << "  --rtt-ms <n>            Round-trip time (default 80)\n"
        << "  --jitter-ms <n>         One-way jitter range (default 12)\n"
        << "  --audio-bps <n>         Encoded realtime audio bitrate (default 32000)\n"
        << "  --audio-frame-ms <n>    Realtime frame cadence (default 20)\n"
        << "  --ttl-ms <n>            Realtime frame deadline (default 180)\n"
        << "  --data-kb <n>           Reliable side payload (default 256)\n"
        << "  --copies <n>            Proactive realtime copies per frame (default 2)\n"
        << "  --seed <n>              RNG seed (default 42)\n"
        << "  -h, --help              Show this help\n";
}

static bool parse_args(int argc, char **argv, SimArgs &args)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](int &v) -> bool {
            if (i + 1 >= argc) return false;
            return parse_int(argv[++i], v);
        };

        if (a == "--duration-ms") {
            if (!need_value(args.duration_ms)) return false;
        } else if (a == "--bandwidth-kbps") {
            if (!need_value(args.bandwidth_kbps)) return false;
        } else if (a == "--loss") {
            if (i + 1 >= argc || !parse_double(argv[++i], args.loss)) return false;
        } else if (a == "--rtt-ms") {
            if (!need_value(args.rtt_ms)) return false;
        } else if (a == "--jitter-ms") {
            if (!need_value(args.jitter_ms)) return false;
        } else if (a == "--audio-bps") {
            if (!need_value(args.audio_bps)) return false;
        } else if (a == "--audio-frame-ms") {
            if (!need_value(args.audio_frame_ms)) return false;
        } else if (a == "--ttl-ms") {
            if (!need_value(args.realtime_ttl_ms)) return false;
        } else if (a == "--data-kb") {
            if (!need_value(args.data_kb)) return false;
        } else if (a == "--copies") {
            if (!need_value(args.realtime_copies)) return false;
        } else if (a == "--seed") {
            int seed = 0;
            if (!need_value(seed)) return false;
            args.seed = static_cast<unsigned>(seed);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (args.duration_ms <= 0 || args.tick_ms <= 0 ||
        args.bandwidth_kbps <= 0 || args.audio_bps <= 0 ||
        args.audio_frame_ms <= 0 || args.realtime_ttl_ms <= 0 ||
        args.data_kb < 0 || args.reliable_chunk_bytes <= 0 ||
        args.realtime_copies <= 0 || args.loss < 0.0 || args.loss > 1.0) {
        return false;
    }
    return true;
}

static double percentile(std::vector<int> values, double p)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double pos = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return static_cast<double>(values[lo]) * (1.0 - frac) +
           static_cast<double>(values[hi]) * frac;
}

class AoqSimulator {
public:
    explicit AoqSimulator(const SimArgs &args)
        : args_(args),
          rng_(args.seed),
          loss_(args.loss),
          jitter_(-args.jitter_ms, args.jitter_ms)
    {
        const int frame_bytes =
            std::max(1, args_.audio_bps * args_.audio_frame_ms / 8000);
        const int frame_count =
            (args_.duration_ms + args_.audio_frame_ms - 1) / args_.audio_frame_ms;
        realtime_.reserve(static_cast<size_t>(frame_count));
        for (int i = 0; i < frame_count; ++i) {
            realtime_.push_back({static_cast<uint64_t>(i),
                                 i * args_.audio_frame_ms,
                                 frame_bytes,
                                 false,
                                 false,
                                 false,
                                 -1});
        }

        int data_bytes = args_.data_kb * 1024;
        uint64_t seq = 0;
        while (data_bytes > 0) {
            int n = std::min(args_.reliable_chunk_bytes, data_bytes);
            reliable_.push_back({seq++, n, false, 0, -1, 0});
            data_bytes -= n;
        }
    }

    void Run()
    {
        const int end_ms = args_.duration_ms + args_.rtt_ms * 8 +
                           args_.realtime_ttl_ms;
        for (now_ms_ = 0; now_ms_ <= end_ms; now_ms_ += args_.tick_ms) {
            ProcessEvents();
            ExpireRealtime();

            double budget = static_cast<double>(args_.bandwidth_kbps) * 1000.0 /
                            8.0 * static_cast<double>(args_.tick_ms) / 1000.0;
            double realtime_budget = budget * 0.75;
            double reliable_budget = budget - realtime_budget;

            SendRealtime(realtime_budget);
            reliable_budget += realtime_budget;
            SendReliable(reliable_budget);

            if (AllDone()) break;
        }
        ProcessEvents();
    }

    void PrintReport() const
    {
        std::vector<int> rt_delays;
        int rt_delivered = 0;
        int rt_expired = 0;
        for (const auto &f : realtime_) {
            if (f.delivered) {
                ++rt_delivered;
                rt_delays.push_back(f.delivered_ms - f.generated_ms);
            } else if (f.expired) {
                ++rt_expired;
            }
        }

        int rel_delivered = 0;
        int rel_last_ms = 0;
        int rel_bytes = 0;
        for (const auto &c : reliable_) {
            if (c.delivered) {
                ++rel_delivered;
                rel_bytes += c.size_bytes;
                rel_last_ms = std::max(rel_last_ms, c.delivered_ms);
            }
        }

        const double rt_loss_pct =
            realtime_.empty() ? 0.0 :
            100.0 * static_cast<double>(realtime_.size() - rt_delivered) /
            static_cast<double>(realtime_.size());
        const double rel_done_pct =
            reliable_.empty() ? 100.0 :
            100.0 * static_cast<double>(rel_delivered) /
            static_cast<double>(reliable_.size());

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "AOQ-like transport simulation\n";
        std::cout << "  link: loss=" << args_.loss * 100.0 << "%, rtt="
                  << args_.rtt_ms << " ms, bandwidth="
                  << args_.bandwidth_kbps << " kbps\n";
        std::cout << "  realtime: generated=" << realtime_.size()
                  << ", delivered=" << rt_delivered
                  << ", expired=" << rt_expired
                  << ", end-to-end loss=" << rt_loss_pct << "%\n";
        std::cout << "  realtime delay: p50=" << percentile(rt_delays, 50.0)
                  << " ms, p95=" << percentile(rt_delays, 95.0)
                  << " ms, p99=" << percentile(rt_delays, 99.0) << " ms\n";
        std::cout << "  reliable: chunks=" << reliable_.size()
                  << ", delivered=" << rel_delivered
                  << " (" << rel_done_pct << "%), bytes=" << rel_bytes
                  << ", complete_at=" << rel_last_ms << " ms\n";
        std::cout << "  packets: realtime_sent=" << stats_.realtime_packets
                  << " lost=" << stats_.realtime_packet_loss
                  << ", reliable_sent=" << stats_.reliable_packets
                  << " lost=" << stats_.reliable_packet_loss
                  << ", reliable_retx=" << stats_.reliable_retransmits << "\n";
        std::cout << "  bytes: realtime=" << stats_.realtime_bytes
                  << ", reliable=" << stats_.reliable_bytes << "\n";
    }

private:
    void ProcessEvents()
    {
        for (auto &e : events_) {
            if (e.at_ms > now_ms_) continue;
            if (e.kind == DeliveryEvent::Realtime) {
                auto &f = realtime_[static_cast<size_t>(e.seq)];
                if (!f.expired && !f.delivered &&
                    e.at_ms <= f.generated_ms + args_.realtime_ttl_ms) {
                    f.delivered = true;
                    f.delivered_ms = e.at_ms;
                }
            } else {
                auto &c = reliable_[static_cast<size_t>(e.seq)];
                if (!c.delivered) {
                    c.delivered = true;
                    c.delivered_ms = e.at_ms;
                }
            }
            e.at_ms = -1;
        }
        events_.erase(std::remove_if(events_.begin(), events_.end(),
                                     [](const DeliveryEvent &e) {
                                         return e.at_ms < 0;
                                     }),
                      events_.end());
    }

    void ExpireRealtime()
    {
        for (auto &f : realtime_) {
            if (!f.delivered && !f.expired &&
                now_ms_ > f.generated_ms + args_.realtime_ttl_ms) {
                f.expired = true;
            }
        }
    }

    void SendRealtime(double &budget)
    {
        for (auto &f : realtime_) {
            if (budget < f.size_bytes || f.sent || f.delivered || f.expired ||
                f.generated_ms > now_ms_) {
                continue;
            }

            f.sent = true;
            const int copies = AdaptRealtimeCopies();
            for (int i = 0; i < copies; ++i) {
                ++stats_.realtime_packets;
                stats_.realtime_bytes += static_cast<uint64_t>(f.size_bytes);
                budget -= static_cast<double>(f.size_bytes);
                if (loss_(rng_)) {
                    ++stats_.realtime_packet_loss;
                } else {
                    Schedule(DeliveryEvent::Realtime, f.seq);
                }
                if (budget < f.size_bytes) break;
            }
            if (budget < f.size_bytes) break;
        }
    }

    void SendReliable(double &budget)
    {
        for (auto &c : reliable_) {
            if (budget < c.size_bytes || c.delivered || c.next_send_ms > now_ms_) {
                continue;
            }

            ++c.sends;
            if (c.sends > 1) ++stats_.reliable_retransmits;
            ++stats_.reliable_packets;
            stats_.reliable_bytes += static_cast<uint64_t>(c.size_bytes);
            budget -= static_cast<double>(c.size_bytes);

            if (loss_(rng_)) {
                ++stats_.reliable_packet_loss;
                c.next_send_ms = now_ms_ + args_.rtt_ms;
            } else {
                Schedule(DeliveryEvent::Reliable, c.seq);
                c.next_send_ms = now_ms_ + args_.rtt_ms * 2;
            }
        }
    }

    void Schedule(DeliveryEvent::Kind kind, uint64_t seq)
    {
        int one_way = std::max(1, args_.rtt_ms / 2 + jitter_(rng_));
        events_.push_back({kind, seq, now_ms_ + one_way});
    }

    int AdaptRealtimeCopies() const
    {
        if (args_.loss >= 0.50) return std::max(args_.realtime_copies, 4);
        if (args_.loss >= 0.30) return std::max(args_.realtime_copies, 3);
        return args_.realtime_copies;
    }

    bool AllDone() const
    {
        const bool rt_done = std::all_of(realtime_.begin(), realtime_.end(),
                                         [](const RealtimeFrame &f) {
                                             return f.delivered || f.expired;
                                         });
        const bool rel_done = std::all_of(reliable_.begin(), reliable_.end(),
                                          [](const ReliableChunk &c) {
                                              return c.delivered;
                                          });
        return rt_done && rel_done && events_.empty();
    }

    SimArgs args_;
    std::vector<RealtimeFrame> realtime_;
    std::vector<ReliableChunk> reliable_;
    std::vector<DeliveryEvent> events_;
    Stats stats_;
    int now_ms_ = 0;
    std::mt19937 rng_;
    std::bernoulli_distribution loss_;
    std::uniform_int_distribution<int> jitter_;
};

int main(int argc, char **argv)
{
    SimArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    AoqSimulator sim(args);
    sim.Run();
    sim.PrintReport();
    return 0;
}
