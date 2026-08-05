#include "rs_xquic_transport.h"

#ifdef RS_ENABLE_XQUIC

#include <algorithm>
#include <cstring>

namespace rs::realtime
{
namespace
{

xqc_data_qos_level_t qos_for_track(TrackType type)
{
    switch (type) {
    case TrackType::Audio: return XQC_DATA_QOS_HIGHEST;
    case TrackType::Video: return XQC_DATA_QOS_HIGH;
    case TrackType::Data: return XQC_DATA_QOS_NORMAL;
    }
    return XQC_DATA_QOS_NORMAL;
}

uint32_t read_be32(const uint8_t *data)
{
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

} // namespace

XquicTransport::XquicTransport(xqc_engine_t *engine, xqc_connection_t *connection,
                               const xqc_cid_t *cid)
    : engine_(engine), connection_(connection)
{
    if (cid) cid_ = *cid;
}

TransportKind XquicTransport::kind() const
{
    return TransportKind::Xquic;
}

bool XquicTransport::connect()
{
    if (!engine_ || !connection_) return false;
    xqc_datagram_set_user_data(connection_, this);
    connected_ = true;
    return true;
}

void XquicTransport::close()
{
    connected_ = false;
    reliable_pending_.clear();
    reliable_receive_buffer_.clear();
}

bool XquicTransport::send_datagram(const MediaObject &object)
{
    if (!connected_ || !connection_) return false;
    const std::vector<uint8_t> encoded = encode_object(object);
    if (encoded.empty()) {
        notify_error("XQUIC object encoding failed");
        return false;
    }
    if (datagram_mss_ != 0 && encoded.size() > datagram_mss_) {
        notify_error("XQUIC DATAGRAM object exceeds negotiated MSS");
        return false;
    }

    uint64_t dgram_id = 0;
    const xqc_int_t ret = xqc_datagram_send(
        connection_, const_cast<uint8_t *>(encoded.data()), encoded.size(),
        &dgram_id, qos_for_track(object.header.track_type));
    if (ret != 0) return false;

    ++stats_.packets_sent;
    stats_.bytes_sent += encoded.size();
    return true;
}

bool XquicTransport::send_reliable(const MediaObject &object)
{
    if (!connected_ || !reliable_stream_) return false;
    const std::vector<uint8_t> encoded = encode_object(object);
    if (encoded.empty()) {
        notify_error("XQUIC reliable object encoding failed");
        return false;
    }
    reliable_pending_.push_back(encoded);
    return flush_reliable();
}

TransportStats XquicTransport::stats() const
{
    TransportStats result = stats_;
    if (engine_ && connection_) {
        const xqc_conn_stats_t xstats = xqc_conn_get_stats(engine_, &cid_);
        result.rtt_us = xstats.srtt;
        result.packets_lost = xstats.lost_count;
        result.packets_sent = xstats.send_count;
        result.packets_received = xstats.recv_count;
    }
    return result;
}

void XquicTransport::set_observer(TransportObserver *observer)
{
    observer_ = observer;
}

void XquicTransport::set_reliable_stream(xqc_stream_t *stream)
{
    reliable_stream_ = stream;
    if (reliable_stream_) xqc_stream_set_user_data(reliable_stream_, this);
}

bool XquicTransport::flush_reliable()
{
    if (!reliable_stream_) return false;
    while (!reliable_pending_.empty()) {
        std::vector<uint8_t> &buffer = reliable_pending_.front();
        if (buffer.empty()) {
            reliable_pending_.pop_front();
            continue;
        }

        const ssize_t sent = xqc_stream_send(
            reliable_stream_, buffer.data(), buffer.size(), 0);
        if (sent < 0) return false;
        if (sent == 0) return false;
        if (static_cast<size_t>(sent) < buffer.size()) {
            buffer.erase(buffer.begin(), buffer.begin() + sent);
            return true;
        }
        stats_.bytes_sent += buffer.size();
        ++stats_.packets_sent;
        reliable_pending_.pop_front();
    }
    return true;
}

bool XquicTransport::receive_reliable(const uint8_t *data, size_t size)
{
    if (!data || size == 0) return true;
    reliable_receive_buffer_.insert(reliable_receive_buffer_.end(), data, data + size);

    while (reliable_receive_buffer_.size() >= k_object_header_size) {
        const uint32_t payload_length = read_be32(reliable_receive_buffer_.data() + 49);
        const size_t object_size = k_object_header_size + payload_length;
        if (object_size > reliable_receive_buffer_.size()) break;

        const DecodeResult decoded = decode_object(reliable_receive_buffer_.data(), object_size);
        if (!decoded.ok) {
            notify_error(decoded.error);
            reliable_receive_buffer_.clear();
            return false;
        }
        reliable_receive_buffer_.erase(
            reliable_receive_buffer_.begin(),
            reliable_receive_buffer_.begin() + static_cast<std::ptrdiff_t>(object_size));
        ++stats_.packets_received;
        stats_.bytes_received += object_size;
        if (observer_) observer_->on_transport_object(decoded.object);
    }
    return true;
}

void XquicTransport::configure_callbacks(xqc_app_proto_callbacks_t &callbacks)
{
    callbacks.dgram_cbs.datagram_read_notify = &XquicTransport::on_datagram_read;
    callbacks.dgram_cbs.datagram_write_notify = &XquicTransport::on_datagram_write;
    callbacks.dgram_cbs.datagram_lost_notify = &XquicTransport::on_datagram_lost;
    callbacks.dgram_cbs.datagram_acked_notify = &XquicTransport::on_datagram_acked;
    callbacks.dgram_cbs.datagram_mss_updated_notify = &XquicTransport::on_datagram_mss_updated;
    callbacks.stream_cbs.stream_read_notify = &XquicTransport::on_stream_read;
    callbacks.stream_cbs.stream_write_notify = &XquicTransport::on_stream_write;
}

void XquicTransport::on_datagram_read(xqc_connection_t *, void *user_data,
                                      const void *data, size_t size, uint64_t)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    if (transport) transport->receive_datagram(static_cast<const uint8_t *>(data), size);
}

void XquicTransport::on_datagram_write(xqc_connection_t *, void *user_data)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    if (transport) transport->flush_reliable();
}

void XquicTransport::on_datagram_lost(xqc_connection_t *, uint64_t, void *user_data)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    if (transport) ++transport->stats_.packets_lost;
}

void XquicTransport::on_datagram_acked(xqc_connection_t *, uint64_t, void *)
{
}

void XquicTransport::on_datagram_mss_updated(xqc_connection_t *, size_t mss, void *user_data)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    if (transport) transport->datagram_mss_ = mss;
}

xqc_int_t XquicTransport::on_stream_read(xqc_stream_t *stream, void *user_data)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    if (!transport) return -1;

    std::vector<uint8_t> buffer(16 * 1024);
    uint8_t fin = 0;
    while (true) {
        const ssize_t received = xqc_stream_recv(stream, buffer.data(), buffer.size(), &fin);
        if (received <= 0) break;
        if (!transport->receive_reliable(buffer.data(), static_cast<size_t>(received))) return -1;
    }
    return 0;
}

xqc_int_t XquicTransport::on_stream_write(xqc_stream_t *, void *user_data)
{
    auto *transport = static_cast<XquicTransport *>(user_data);
    return transport && transport->flush_reliable() ? 0 : -1;
}

void XquicTransport::notify_error(const std::string &message)
{
    if (observer_) observer_->on_transport_error(message);
}

bool XquicTransport::receive_datagram(const uint8_t *data, size_t size)
{
    const DecodeResult decoded = decode_object(data, size);
    if (!decoded.ok) {
        notify_error(decoded.error);
        return false;
    }
    ++stats_.packets_received;
    stats_.bytes_received += size;
    if (observer_) observer_->on_transport_object(decoded.object);
    return true;
}

} // namespace rs::realtime

#endif // RS_ENABLE_XQUIC
