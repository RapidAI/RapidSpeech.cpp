#pragma once

#include "rs_realtime_protocol.h"

#ifdef RS_ENABLE_XQUIC

#include <deque>
#include <vector>

#include <xquic/xquic.h>

namespace rs::realtime
{

class XquicTransport final : public RealtimeTransport
{
public:
    XquicTransport(xqc_engine_t *engine, xqc_connection_t *connection,
                   const xqc_cid_t *cid);

    TransportKind kind() const override;
    bool connect() override;
    void close() override;
    bool send_datagram(const MediaObject &object) override;
    bool send_reliable(const MediaObject &object) override;
    TransportStats stats() const override;
    void set_observer(TransportObserver *observer) override;

    void set_reliable_stream(xqc_stream_t *stream);
    bool flush_reliable();
    bool receive_reliable(const uint8_t *data, size_t size);

    static void configure_callbacks(xqc_app_proto_callbacks_t &callbacks);

private:
    static void on_datagram_read(xqc_connection_t *connection, void *user_data,
                                 const void *data, size_t size, uint64_t unix_ts);
    static void on_datagram_write(xqc_connection_t *connection, void *user_data);
    static void on_datagram_lost(xqc_connection_t *connection, uint64_t dgram_id,
                                 void *user_data);
    static void on_datagram_acked(xqc_connection_t *connection, uint64_t dgram_id,
                                  void *user_data);
    static void on_datagram_mss_updated(xqc_connection_t *connection, size_t mss,
                                         void *user_data);
    static xqc_int_t on_stream_read(xqc_stream_t *stream, void *user_data);
    static xqc_int_t on_stream_write(xqc_stream_t *stream, void *user_data);

    void notify_error(const std::string &message);
    bool receive_datagram(const uint8_t *data, size_t size);

    xqc_engine_t *engine_ = nullptr;
    xqc_connection_t *connection_ = nullptr;
    xqc_cid_t cid_{};
    xqc_stream_t *reliable_stream_ = nullptr;
    TransportObserver *observer_ = nullptr;
    std::deque<std::vector<uint8_t>> reliable_pending_;
    std::vector<uint8_t> reliable_receive_buffer_;
    TransportStats stats_;
    size_t datagram_mss_ = 0;
    bool connected_ = false;
};

} // namespace rs::realtime

#endif // RS_ENABLE_XQUIC
