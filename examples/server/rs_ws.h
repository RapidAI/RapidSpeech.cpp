// rs_ws.h — a minimal RFC 6455 WebSocket server (server side only).
//
// Just enough for the streaming ASR/TTS endpoints: HTTP upgrade handshake
// (SHA-1 + base64 of Sec-WebSocket-Key), frame read with unmasking and
// continuation reassembly, frame write (text/binary/close/ping/pong). No TLS,
// no permessage-deflate, no client role. Thread-per-connection.
//
// Usage:
//   rsws::Server ws;
//   ws.route("/asr/stream", [](rsws::Conn &c) {
//       int op; std::string msg;
//       while (c.recv(op, msg)) {
//           if (op == rsws::OP_BINARY) { ... audio ... }
//           else if (op == rsws::OP_TEXT) { ... json control ... }
//           c.send_text("{\"type\":\"partial\",...}");
//       }
//   });
//   ws.listen("127.0.0.1", 8081);   // blocks

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using rs_socket_t = SOCKET;
#define RS_INVALID_SOCKET INVALID_SOCKET
#define RS_CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using rs_socket_t = int;
#define RS_INVALID_SOCKET (-1)
#define RS_CLOSESOCK ::close
#endif

namespace rsws {

enum { OP_CONT = 0x0, OP_TEXT = 0x1, OP_BINARY = 0x2, OP_CLOSE = 0x8,
       OP_PING = 0x9, OP_PONG = 0xA };

// ── SHA-1 (public-domain style, compact) ───────────────────────────────
namespace detail {
struct Sha1 {
    uint32_t h[5];
    uint64_t len = 0;
    unsigned char buf[64];
    int bidx = 0;
    Sha1() { h[0]=0x67452301; h[1]=0xEFCDAB89; h[2]=0x98BADCFE; h[3]=0x10325476; h[4]=0xC3D2E1F0; }
    static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }
    void block(const unsigned char *p) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
        for (int i = 16; i < 80; i++) w[i] = rol(w[i-3]^w[i-8]^w[i-14]^w[i-16], 1);
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b&c)|(b&d)|(c&d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t t = rol(a,5) + f + e + k + w[i];
            e = d; d = c; c = rol(b,30); b = a; a = t;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e;
    }
    void add(const void *data, size_t n) {
        const unsigned char *p = (const unsigned char *)data;
        len += n * 8;
        while (n--) { buf[bidx++] = *p++; if (bidx == 64) { block(buf); bidx = 0; } }
    }
    void finish(unsigned char out[20]) {
        uint64_t bits = len; // original message length in bits (before padding)
        unsigned char pad = 0x80; add(&pad, 1);
        unsigned char z = 0;
        while (bidx != 56) add(&z, 1);
        unsigned char lb[8];
        for (int i = 7; i >= 0; i--) { lb[i] = bits & 0xFF; bits >>= 8; }
        add(lb, 8);
        for (int i = 0; i < 5; i++) {
            out[i*4] = h[i]>>24; out[i*4+1] = h[i]>>16;
            out[i*4+2] = h[i]>>8; out[i*4+3] = h[i];
        }
    }
};

inline std::string base64(const unsigned char *d, size_t n) {
    static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = d[i] << 16;
        if (i + 1 < n) v |= d[i+1] << 8;
        if (i + 2 < n) v |= d[i+2];
        o += t[(v >> 18) & 63];
        o += t[(v >> 12) & 63];
        o += (i + 1 < n) ? t[(v >> 6) & 63] : '=';
        o += (i + 2 < n) ? t[v & 63] : '=';
    }
    return o;
}

inline bool recv_all(rs_socket_t s, void *buf, size_t n) {
    char *p = (char *)buf;
    while (n) {
        int r = (int)recv(s, p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}
inline bool send_all(rs_socket_t s, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        int r = (int)send(s, p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}
} // namespace detail

// ── connection ─────────────────────────────────────────────────────────
class Conn {
public:
    explicit Conn(rs_socket_t s) : sock_(s) {}

    // Receive one complete message (reassembling continuation frames and
    // transparently answering ping/close). Returns false on close/error.
    bool recv(int &opcode, std::string &payload) {
        payload.clear();
        int msg_op = 0;
        for (;;) {
            unsigned char h[2];
            if (!detail::recv_all(sock_, h, 2)) return false;
            bool fin = h[0] & 0x80;
            int op = h[0] & 0x0F;
            bool masked = h[1] & 0x80;
            uint64_t len = h[1] & 0x7F;
            if (len == 126) {
                unsigned char e[2];
                if (!detail::recv_all(sock_, e, 2)) return false;
                len = (e[0] << 8) | e[1];
            } else if (len == 127) {
                unsigned char e[8];
                if (!detail::recv_all(sock_, e, 8)) return false;
                len = 0;
                for (int i = 0; i < 8; i++) len = (len << 8) | e[i];
            }
            unsigned char mask[4] = {0,0,0,0};
            if (masked && !detail::recv_all(sock_, mask, 4)) return false;
            if (len > kMaxMessage) return false; // guard
            std::string frag((size_t)len, '\0');
            if (len && !detail::recv_all(sock_, &frag[0], (size_t)len)) return false;
            if (masked)
                for (size_t i = 0; i < frag.size(); i++) frag[i] ^= mask[i & 3];

            if (op == OP_PING) { send_frame(OP_PONG, frag.data(), frag.size()); continue; }
            if (op == OP_PONG) { continue; }
            if (op == OP_CLOSE) { send_frame(OP_CLOSE, nullptr, 0); return false; }
            if (op != OP_CONT) msg_op = op;
            payload += frag;
            if (fin) { opcode = msg_op; return true; }
        }
    }

    bool send_text(const std::string &s) { return send_frame(OP_TEXT, s.data(), s.size()); }
    bool send_binary(const void *d, size_t n) { return send_frame(OP_BINARY, d, n); }
    void close_conn() { send_frame(OP_CLOSE, nullptr, 0); }

    const std::string &path() const { return path_; }
    const std::string &query() const { return query_; }

private:
    friend class Server;
    rs_socket_t sock_;
    std::string path_, query_;
    static constexpr uint64_t kMaxMessage = 64ull * 1024 * 1024;

    bool send_frame(int opcode, const void *data, size_t n) {
        unsigned char h[10];
        size_t hn = 0;
        h[hn++] = 0x80 | (opcode & 0x0F); // FIN + opcode
        if (n < 126) h[hn++] = (unsigned char)n;
        else if (n <= 0xFFFF) {
            h[hn++] = 126; h[hn++] = (n >> 8) & 0xFF; h[hn++] = n & 0xFF;
        } else {
            h[hn++] = 127;
            for (int i = 7; i >= 0; i--) h[hn++] = (n >> (i * 8)) & 0xFF;
        }
        if (!detail::send_all(sock_, h, hn)) return false;
        return n == 0 || detail::send_all(sock_, data, n);
    }
};

// ── server ─────────────────────────────────────────────────────────────
class Server {
public:
    using Handler = std::function<void(Conn &)>;
    void route(const std::string &path, Handler h) { routes_[path] = std::move(h); }

    // Blocks. Returns false if the listen socket could not be bound.
    bool listen(const std::string &host, int port) {
#ifdef _WIN32
        WSADATA w; WSAStartup(MAKEWORD(2, 2), &w);
#endif
        rs_socket_t ls = socket(AF_INET, SOCK_STREAM, 0);
        if (ls == RS_INVALID_SOCKET) return false;
        int yes = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        if (bind(ls, (sockaddr *)&addr, sizeof(addr)) != 0) { RS_CLOSESOCK(ls); return false; }
        // Backlog large enough for bursty connection acceptance: the previous
        // value (16) caused simultaneous-connect bursts > ~22 to get RST'd
        // before accept() could drain them (seen as "no close frame received
        // or sent" on the client). 512 comfortably absorbs a connection storm;
        // the kernel clamps to somaxconn automatically.
        if (::listen(ls, 512) != 0) { RS_CLOSESOCK(ls); return false; }
        for (;;) {
            rs_socket_t cs = accept(ls, nullptr, nullptr);
            if (cs == RS_INVALID_SOCKET) continue;
            std::thread(&Server::serve, this, cs).detach();
        }
        return true;
    }

private:
    std::map<std::string, Handler> routes_;

    void serve(rs_socket_t cs) {
        int one = 1;
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
        Conn conn(cs);
        if (handshake(conn)) {
            auto it = routes_.find(conn.path());
            if (it != routes_.end()) {
                try { it->second(conn); } catch (...) {}
            } else {
                conn.send_text("{\"type\":\"error\",\"error\":\"unknown path\"}");
            }
        }
        conn.close_conn();
        RS_CLOSESOCK(cs);
    }

    // Read the HTTP upgrade request, validate, reply 101.
    bool handshake(Conn &conn) {
        std::string req;
        char c;
        while (req.find("\r\n\r\n") == std::string::npos) {
            int r = (int)recv(conn.sock_, &c, 1, 0);
            if (r <= 0) return false;
            req += c;
            if (req.size() > 16384) return false;
        }
        // request line: GET /path?query HTTP/1.1
        size_t sp1 = req.find(' ');
        size_t sp2 = req.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
        std::string target = req.substr(sp1 + 1, sp2 - sp1 - 1);
        size_t q = target.find('?');
        if (q == std::string::npos) conn.path_ = target;
        else { conn.path_ = target.substr(0, q); conn.query_ = target.substr(q + 1); }

        std::string key = header(req, "sec-websocket-key");
        if (key.empty()) {
            const char *bad = "HTTP/1.1 400 Bad Request\r\n\r\n";
            detail::send_all(conn.sock_, bad, strlen(bad));
            return false;
        }
        std::string accept_src = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        detail::Sha1 sha;
        sha.add(accept_src.data(), accept_src.size());
        unsigned char digest[20];
        sha.finish(digest);
        std::string accept = detail::base64(digest, 20);
        std::string resp =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
        return detail::send_all(conn.sock_, resp.data(), resp.size());
    }

    static std::string header(const std::string &req, const std::string &name) {
        // case-insensitive header lookup
        std::string lower = req;
        for (char &ch : lower) ch = (char)tolower((unsigned char)ch);
        std::string key = "\r\n" + name + ":";
        size_t p = lower.find(key);
        if (p == std::string::npos) return "";
        p += key.size();
        size_t e = req.find("\r\n", p);
        std::string v = req.substr(p, e - p);
        size_t s = v.find_first_not_of(" \t");
        size_t t = v.find_last_not_of(" \t\r");
        return (s == std::string::npos) ? "" : v.substr(s, t - s + 1);
    }
};

} // namespace rsws
