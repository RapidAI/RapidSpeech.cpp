// kokoro_zh_tone_sandhi.cpp — port of misaki/tone_sandhi.py.
//
// All char indexing is by UTF-8 codepoint, mirroring Python str indexing.

#include "frontend/kokoro_zh_tone_sandhi.h"
#include "frontend/kokoro_pinyin.h"

#include "cppjieba/Jieba.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace rs::kokoro_zh {

// -------------------- utility helpers --------------------

static std::string utf8_last_n_chars(const std::string& s, size_t n) {
    size_t total = utf8_char_count(s);
    if (n >= total) return s;
    return utf8_substr(s, total - n, total);
}

// True iff all characters in `s` are ASCII digits (Python str.isnumeric()
// is broader but the call sites here only deal with Arabic digits).
static bool is_all_ascii_digit(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!(c >= '0' && c <= '9')) return false;
    return true;
}

// Mirrors Python's str.isnumeric() for the subset that matters in zh sandhi:
// ASCII digits + Chinese numerals (零〇一二三四五六七八九十百千万亿 + 两 +
// fullwidth digits). Used by the "_neural_sandhi" '个' check so e.g. "一个"
// triggers ge→ge5.
static bool is_numeric_like(const std::string& s) {
    if (s.empty()) return false;
    if (is_all_ascii_digit(s)) return true;
    static const std::unordered_set<std::string> kNumChars{
        "零","〇","一","二","三","四","五","六","七","八","九","十","百","千","万","亿","兩","两",
        "壹","贰","叁","肆","伍","陆","柒","捌","玖","拾","佰","仟",
        "０","１","2","３","４","５","６","７","８","９"
    };
    return kNumChars.count(s) > 0;
}

static bool find_in_set(const std::unordered_set<std::string>& set,
                        const std::string& s) {
    return set.find(s) != set.end();
}

// Set last digit of a TONE3 syllable to `tone`. If empty or no trailing
// digit, just appends.
static std::string retone(const std::string& s, char tone) {
    if (s.empty()) return std::string(1, tone);
    if (s.back() >= '0' && s.back() <= '9') {
        std::string r = s;
        r.back() = tone;
        return r;
    }
    return s + tone;
}

static bool ends_with_tone(const std::string& s, char tone) {
    return !s.empty() && s.back() == tone;
}

// -------------------- punctuation / sandhi sets --------------------

ToneSandhi::ToneSandhi() {
    // must_neural_tone_words (tone_sandhi.py:31-70).
    static const char* kMustNeural[] = {
        "麻烦","麻利","鸳鸯","高粱","骨头","骆驼","马虎","首饰","馒头","馄饨","风筝",
        "难为","队伍","阔气","闺女","门道","锄头","铺盖","铃铛","铁匠","钥匙","里脊",
        "里头","部分","那么","道士","造化","迷糊","连累","这么","这个","运气","过去",
        "软和","转悠","踏实","跳蚤","跟头","趔趄","财主","豆腐","讲究","记性","记号",
        "认识","规矩","见识","裁缝","补丁","衣裳","衣服","衙门","街坊","行李","行当",
        "蛤蟆","蘑菇","薄荷","葫芦","葡萄","萝卜","荸荠","苗条","苗头","苍蝇","芝麻",
        "舒服","舒坦","舌头","自在","膏药","脾气","脑袋","脊梁","能耐","胳膊","胭脂",
        "胡萝","胡琴","胡同","聪明","耽误","耽搁","耷拉","耳朵","老爷","老实","老婆",
        "戏弄","将军","翻腾","罗嗦","罐头","编辑","结实","红火","累赘","糨糊","糊涂",
        "精神","粮食","簸箕","篱笆","算计","算盘","答应","笤帚","笑语","笑话","窟窿",
        "窝囊","窗户","稳当","稀罕","称呼","秧歌","秀气","秀才","福气","祖宗","砚台",
        "码头","石榴","石头","石匠","知识","眼睛","眯缝","眨巴","眉毛","相声","盘算",
        "白净","痢疾","痛快","疟疾","疙瘩","疏忽","畜生","生意","甘蔗","琵琶","琢磨",
        "琉璃","玻璃","玫瑰","玄乎","狐狸","状元","特务","牲口","牙碜","牌楼","爽快",
        "爱人","热闹","烧饼","烟筒","烂糊","点心","炊帚","灯笼","火候","漂亮","滑溜",
        "溜达","温和","清楚","消息","浪头","活泼","比方","正经","欺负","模糊","槟榔",
        "棺材","棒槌","棉花","核桃","栅栏","柴火","架势","枕头","枇杷","机灵","本事",
        "木头","木匠","朋友","月饼","月亮","暖和","明白","时候","新鲜","故事","收拾",
        "收成","提防","挖苦","挑剔","指甲","指头","拾掇","拳头","拨弄","招牌","招呼",
        "抬举","护士","折腾","扫帚","打量","打算","打扮","打听","打发","扎实","扁担",
        "戒指","懒得","意识","意思","悟性","怪物","思量","怎么","念头","念叨","别人",
        "快活","忙活","志气","心思","得罪","张罗","弟兄","开通","应酬","庄稼","干事",
        "帮手","帐篷","希罕","师父","师傅","巴结","巴掌","差事","工夫","岁数","屁股",
        "尾巴","少爷","小气","小伙","将就","对头","对付","寡妇","家伙","客气","实在",
        "官司","学问","字号","嫁妆","媳妇","媒人","婆家","娘家","委屈","姑娘","姐夫",
        "妯娌","妥当","妖精","奴才","女婿","头发","太阳","大爷","大方","大意","大夫",
        "多少","多么","外甥","壮实","地道","地方","在乎","困难","嘴巴","嘱咐","嘟囔",
        "嘀咕","喜欢","喇嘛","喇叭","商量","唾沫","哑巴","哈欠","哆嗦","咳嗽","和尚",
        "告诉","告示","含糊","吓唬","后头","名字","名堂","合同","吆喝","叫唤","口袋",
        "厚道","厉害","千斤","包袱","包涵","匀称","勤快","动静","动弹","功夫","力气",
        "前头","刺猬","刺激","别扭","利落","利索","利害","分析","出息","凑合","凉快",
        "冷战","冤枉","冒失","养活","关系","先生","兄弟","便宜","使唤","佩服","作坊",
        "体面","位置","似的","伙计","休息","什么","人家","亲戚","亲家","交情","云彩",
        "事情","买卖","主意","丫头","丧气","两口","东西","东家","世故","不由","下水",
        "下巴","上头","上司","丈夫","丈人","一辈","那个","菩萨","父亲","母亲","咕噜",
        "邋遢","费用","冤家","甜头","介绍","荒唐","大人","泥鳅","幸福","熟悉","计划",
        "扑腾","蜡烛","姥爷","照顾","喉咙","吉他","弄堂","蚂蚱","凤凰","拖沓","寒碜",
        "糟蹋","倒腾","报复","逻辑","盘缠","喽啰","牢骚","咖喱","扫把","惦记"
    };
    for (auto* w : kMustNeural) must_neural_.insert(w);

    static const char* kMustNotNeural[] = {
        "男子","女子","分子","原子","量子","莲子","石子","瓜子","电子","人人","虎虎",
        "幺幺","干嘛","学子","哈哈","数数","袅袅","局地","以下","娃哈哈","花花草草","留得",
        "耕地","想想","熙熙","攘攘","卵子","死死","冉冉","恳恳","佼佼","吵吵","打打",
        "考考","整整","莘莘","落地","算子","家家户户","青青"
    };
    for (auto* w : kMustNotNeural) must_not_neural_.insert(w);

    // self.punc = "、：，；。？！“”‘’':,;.?!" (Python str of single chars).
    // Each codepoint becomes one entry in our set.
    static const char* kPunc[] = {
        "、","：","，","；","。","？","！","“","”","‘","’","'", ":", ",", ";", ".", "?", "!"
    };
    for (auto* p : kPunc) punc_chars_.insert(p);
}

// -------------------- split_word --------------------

std::vector<std::string> ToneSandhi::split_word(const std::string& word) const {
    std::vector<std::string> raw;
    if (jieba_) {
        std::vector<std::string> tmp;
        jieba_->CutForSearch(word, tmp);
        raw = std::move(tmp);
    } else {
        raw.push_back(word);
    }
    // Python: sorted by len ascending, take first as shortest.
    std::stable_sort(raw.begin(), raw.end(),
                     [](const std::string& a, const std::string& b) {
                         return utf8_char_count(a) < utf8_char_count(b);
                     });
    if (raw.empty()) return {word};
    const std::string& first = raw.front();
    // word.find(first) — character index, but std::string::find on UTF-8
    // matches at byte boundaries which align with codepoints by construction
    // since Jieba returns UTF-8 substrings of `word`.
    size_t bpos = word.find(first);
    if (bpos == 0) {
        std::string second = word.substr(first.size());
        return {first, second};
    } else {
        // first appears later; second = word[:-len(first)] by chars
        size_t n_chars_word = utf8_char_count(word);
        size_t n_chars_first = utf8_char_count(first);
        std::string second = utf8_substr(word, 0, n_chars_word - n_chars_first);
        return {second, first};
    }
}

// -------------------- finals_tone3_for --------------------

std::vector<std::string> ToneSandhi::finals_tone3_for(
    const std::string& word) const {
    if (!py_) return {};
    return py_->LazyPinyinFinalsTone3(word);
}

// -------------------- all_tone_three --------------------

bool ToneSandhi::all_tone_three(const std::vector<std::string>& finals) {
    if (finals.empty()) return false;
    for (const auto& f : finals) {
        if (f.empty() || f.back() != '3') return false;
    }
    return true;
}

// -------------------- bu_sandhi --------------------

std::vector<std::string> ToneSandhi::bu_sandhi(
    const std::string& word, std::vector<std::string> finals) const {
    const size_t n = utf8_char_count(word);
    if (n != finals.size()) return finals;
    if (n == 3 && utf8_char_at(word, 1) == "不") {
        finals[1] = retone(finals[1], '5');
    } else {
        for (size_t i = 0; i < n; ++i) {
            std::string ch = utf8_char_at(word, i);
            if (ch == "不" && i + 1 < n
                && !finals[i + 1].empty() && finals[i + 1].back() == '4') {
                finals[i] = retone(finals[i], '2');
            }
        }
    }
    return finals;
}

// -------------------- yi_sandhi --------------------

std::vector<std::string> ToneSandhi::yi_sandhi(
    const std::string& word, std::vector<std::string> finals) const {
    const size_t n = utf8_char_count(word);
    if (n != finals.size()) return finals;

    // "一" in number sequences
    bool has_yi = false;
    bool all_numeric_except_yi = true;
    for (size_t i = 0; i < n; ++i) {
        std::string ch = utf8_char_at(word, i);
        if (ch == "一") { has_yi = true; continue; }
        if (!is_all_ascii_digit(ch)) { all_numeric_except_yi = false; break; }
    }
    if (has_yi && all_numeric_except_yi) return finals;

    if (n == 3 && utf8_char_at(word, 1) == "一"
        && utf8_char_at(word, 0) == utf8_char_at(word, n - 1)) {
        finals[1] = retone(finals[1], '5');
    } else {
        // word.startswith("第一")
        if (n >= 2 && utf8_char_at(word, 0) == "第"
            && utf8_char_at(word, 1) == "一") {
            finals[1] = retone(finals[1], '1');
        } else {
            for (size_t i = 0; i < n; ++i) {
                std::string ch = utf8_char_at(word, i);
                if (ch == "一" && i + 1 < n) {
                    const std::string& nxt = finals[i + 1];
                    char tail = nxt.empty() ? '\0' : nxt.back();
                    if (tail == '4' || tail == '5') {
                        finals[i] = retone(finals[i], '2');
                    } else {
                        std::string nxt_ch = utf8_char_at(word, i + 1);
                        if (!find_in_set(punc_chars_, nxt_ch)) {
                            finals[i] = retone(finals[i], '4');
                        }
                    }
                }
            }
        }
    }
    return finals;
}

// -------------------- neural_sandhi --------------------

std::vector<std::string> ToneSandhi::neural_sandhi(
    const std::string& word, const std::string& pos,
    std::vector<std::string> finals) const {
    if (find_in_set(must_not_neural_, word)) return finals;

    const size_t n = utf8_char_count(word);
    if (finals.size() != n) {
        // Some pre-merged words may produce mismatched finals; emulate
        // python which would crash but we just bail safely.
        return finals;
    }

    // Reduplication: for n. v. a.
    if (!pos.empty()) {
        char head = pos[0];
        if (head == 'n' || head == 'v' || head == 'a') {
            for (size_t j = 1; j < n; ++j) {
                if (utf8_char_at(word, j) == utf8_char_at(word, j - 1)) {
                    finals[j] = retone(finals[j], '5');
                }
            }
        }
    }

    auto find_char = [&](const std::string& ch) -> long {
        for (size_t i = 0; i < n; ++i) {
            if (utf8_char_at(word, i) == ch) return (long)i;
        }
        return -1;
    };

    long ge_idx = find_char("个");

    const std::string last_char = (n >= 1) ? utf8_char_at(word, n - 1) : "";
    const std::string penult    = (n >= 2) ? utf8_char_at(word, n - 2) : "";

    static const std::unordered_set<std::string> kPart1{
        "吧","呢","啊","呐","噻","嘛","吖","嗨","哦","哒","滴","哩","哟","喽","啰","耶","喔","诶"};
    static const std::unordered_set<std::string> kPart2{"的","地","得"};
    static const std::unordered_set<std::string> kLeZheGuo{"了","着","过"};
    static const std::unordered_set<std::string> kMenZi{"们","子"};
    static const std::unordered_set<std::string> kShangXia{"上","下"};
    static const std::unordered_set<std::string> kLaiQu{"来","去"};
    static const std::unordered_set<std::string> kShangXiaJinChuHuiGuoQiKai{
        "上","下","进","出","回","过","起","开"};
    static const std::unordered_set<std::string> kUlUzUg{"ul","uz","ug"};
    static const std::unordered_set<std::string> kSlF{"s","l","f"};
    static const std::unordered_set<std::string> kRn{"r","n"};
    static const std::unordered_set<std::string> kGeContext{
        "几","有","两","半","多","各","整","每","做","是"};

    bool handled = false;
    if (n >= 1 && find_in_set(kPart1, last_char)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if (n >= 1 && find_in_set(kPart2, last_char)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if (n == 1 && find_in_set(kLeZheGuo, word)
               && find_in_set(kUlUzUg, pos)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if (n > 1 && find_in_set(kMenZi, last_char)
               && find_in_set(kRn, pos)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if (n > 1 && find_in_set(kShangXia, last_char)
               && find_in_set(kSlF, pos)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if (n > 1 && find_in_set(kLaiQu, last_char)
               && find_in_set(kShangXiaJinChuHuiGuoQiKai, penult)) {
        finals[n - 1] = retone(finals[n - 1], '5');
        handled = true;
    } else if ((ge_idx >= 1 &&
                (is_numeric_like(utf8_char_at(word, ge_idx - 1))
                 || find_in_set(kGeContext, utf8_char_at(word, ge_idx - 1))))
               || word == "个") {
        finals[ge_idx] = retone(finals[ge_idx], '5');
        handled = true;
    }

    if (!handled) {
        std::string last2 = utf8_last_n_chars(word, 2);
        if (find_in_set(must_neural_, word)
            || find_in_set(must_neural_, last2)) {
            finals[n - 1] = retone(finals[n - 1], '5');
        }
    }

    // Sub-word neural pass via _split_word.
    std::vector<std::string> wl = split_word(word);
    if (wl.size() == 2) {
        size_t first_chars = utf8_char_count(wl[0]);
        if (first_chars <= finals.size()) {
            std::vector<std::string> a(finals.begin(),
                                       finals.begin() + first_chars);
            std::vector<std::string> b(finals.begin() + first_chars,
                                       finals.end());
            std::vector<std::vector<std::string>*> pair = {&a, &b};
            for (size_t i = 0; i < wl.size(); ++i) {
                std::string last2 = utf8_last_n_chars(wl[i], 2);
                if (find_in_set(must_neural_, wl[i])
                    || find_in_set(must_neural_, last2)) {
                    auto& v = *pair[i];
                    if (!v.empty()) v.back() = retone(v.back(), '5');
                }
            }
            finals.clear();
            for (auto& x : a) finals.push_back(std::move(x));
            for (auto& x : b) finals.push_back(std::move(x));
        }
    }
    return finals;
}

// -------------------- three_sandhi --------------------

std::vector<std::string> ToneSandhi::three_sandhi(
    const std::string& word, std::vector<std::string> finals) const {
    const size_t n = utf8_char_count(word);
    if (n == 2 && all_tone_three(finals)) {
        finals[0] = retone(finals[0], '2');
        return finals;
    }
    if (n == 3) {
        std::vector<std::string> wl = split_word(word);
        if (all_tone_three(finals)) {
            if (wl.size() == 2 && utf8_char_count(wl[0]) == 2) {
                finals[0] = retone(finals[0], '2');
                finals[1] = retone(finals[1], '2');
            } else if (wl.size() == 2 && utf8_char_count(wl[0]) == 1) {
                finals[1] = retone(finals[1], '2');
            }
        } else if (wl.size() == 2) {
            size_t k = utf8_char_count(wl[0]);
            if (k > finals.size()) k = finals.size();
            std::vector<std::string> a(finals.begin(), finals.begin() + k);
            std::vector<std::string> b(finals.begin() + k, finals.end());
            std::vector<std::vector<std::string>*> pair = {&a, &b};
            for (size_t i = 0; i < pair.size(); ++i) {
                auto& sub = *pair[i];
                if (sub.size() == 2 && all_tone_three(sub)) {
                    sub[0] = retone(sub[0], '2');
                } else if (i == 1 && !sub.empty() && !all_tone_three(sub)
                           && ends_with_tone(sub[0], '3')
                           && !a.empty() && ends_with_tone(a.back(), '3')) {
                    a.back() = retone(a.back(), '2');
                }
            }
            finals.clear();
            for (auto& x : a) finals.push_back(std::move(x));
            for (auto& x : b) finals.push_back(std::move(x));
        }
    } else if (n == 4 && finals.size() == 4) {
        std::vector<std::string> a{finals[0], finals[1]};
        std::vector<std::string> b{finals[2], finals[3]};
        finals.clear();
        if (all_tone_three(a)) a[0] = retone(a[0], '2');
        if (all_tone_three(b)) b[0] = retone(b[0], '2');
        finals.push_back(std::move(a[0])); finals.push_back(std::move(a[1]));
        finals.push_back(std::move(b[0])); finals.push_back(std::move(b[1]));
    }
    return finals;
}

// -------------------- merge passes --------------------

static const std::unordered_set<std::string> kXEng = {"x", "eng"};

WordPosVec ToneSandhi::merge_bu(const WordPosVec& seg) const {
    WordPosVec out;
    out.reserve(seg.size());
    for (size_t i = 0; i < seg.size(); ++i) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;
        if (!find_in_set(kXEng, pos)) {
            if (i > 0 && seg[i - 1].first == "不") {
                word = std::string("不") + word;
            }
        }
        std::string next_pos;
        bool has_next = i + 1 < seg.size();
        if (has_next) next_pos = seg[i + 1].second;
        if (word != "不" || !has_next || find_in_set(kXEng, next_pos)) {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

WordPosVec ToneSandhi::merge_yi(const WordPosVec& seg_in) const {
    WordPosVec seg = seg_in;
    WordPosVec out;
    out.reserve(seg.size());
    bool skip_next = false;
    for (size_t i = 0; i < seg.size(); ++i) {
        if (skip_next) { skip_next = false; continue; }
        const auto& [word, pos] = seg[i];
        if (i >= 1 && word == "一" && i + 1 < seg.size()
            && seg[i - 1].first == seg[i + 1].first
            && seg[i - 1].second == "v"
            && !find_in_set(kXEng, seg[i + 1].second)) {
            out.back().first = out.back().first + "一" + seg[i + 1].first;
            skip_next = true;
        } else {
            out.emplace_back(word, pos);
        }
    }
    seg = out;
    out.clear();
    for (size_t i = 0; i < seg.size(); ++i) {
        const auto& [word, pos] = seg[i];
        if (!out.empty() && out.back().first == "一"
            && !find_in_set(kXEng, pos)) {
            out.back().first = out.back().first + word;
        } else {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

WordPosVec ToneSandhi::merge_reduplication(
    const WordPosVec& seg) const {
    WordPosVec out;
    out.reserve(seg.size());
    for (const auto& [word, pos] : seg) {
        if (!out.empty() && word == out.back().first
            && !find_in_set(kXEng, pos)) {
            out.back().first = out.back().first + word;
        } else {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

static bool is_reduplication(const std::string& w) {
    return utf8_char_count(w) == 2
           && utf8_char_at(w, 0) == utf8_char_at(w, 1);
}

WordPosVec ToneSandhi::merge_continuous_three_tones(
    const WordPosVec& seg) const {
    WordPosVec out;
    std::vector<std::vector<std::string>> sub_finals;
    sub_finals.reserve(seg.size());
    for (const auto& [word, pos] : seg) {
        if (find_in_set(kXEng, pos)) {
            sub_finals.push_back({"0"});
        } else {
            sub_finals.push_back(finals_tone3_for(word));
        }
    }
    std::vector<bool> merge_last(seg.size(), false);
    for (size_t i = 0; i < seg.size(); ++i) {
        const auto& [word, pos] = seg[i];
        if (!find_in_set(kXEng, pos) && i >= 1
            && all_tone_three(sub_finals[i - 1])
            && all_tone_three(sub_finals[i])
            && !merge_last[i - 1]) {
            if (!is_reduplication(seg[i - 1].first)
                && utf8_char_count(seg[i - 1].first) + utf8_char_count(seg[i].first) <= 3) {
                out.back().first += seg[i].first;
                merge_last[i] = true;
            } else {
                out.emplace_back(word, pos);
            }
        } else {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

WordPosVec ToneSandhi::merge_continuous_three_tones_2(
    const WordPosVec& seg) const {
    WordPosVec out;
    std::vector<std::vector<std::string>> sub_finals;
    sub_finals.reserve(seg.size());
    for (const auto& [word, pos] : seg) {
        if (find_in_set(kXEng, pos)) {
            sub_finals.push_back({"0"});
        } else {
            sub_finals.push_back(finals_tone3_for(word));
        }
    }
    std::vector<bool> merge_last(seg.size(), false);
    for (size_t i = 0; i < seg.size(); ++i) {
        const auto& [word, pos] = seg[i];
        const auto& prev = (i >= 1) ? sub_finals[i - 1] : std::vector<std::string>{};
        const auto& cur = sub_finals[i];
        bool prev_ends3 = !prev.empty() && !prev.back().empty() && prev.back().back() == '3';
        bool cur_starts3 = !cur.empty() && !cur.front().empty() && cur.front().back() == '3';
        if (!find_in_set(kXEng, pos) && i >= 1 && prev_ends3 && cur_starts3
            && !merge_last[i - 1]) {
            if (!is_reduplication(seg[i - 1].first)
                && utf8_char_count(seg[i - 1].first) + utf8_char_count(seg[i].first) <= 3) {
                out.back().first += seg[i].first;
                merge_last[i] = true;
            } else {
                out.emplace_back(word, pos);
            }
        } else {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

WordPosVec ToneSandhi::merge_er(const WordPosVec& seg) const {
    WordPosVec out;
    for (size_t i = 0; i < seg.size(); ++i) {
        const auto& [word, pos] = seg[i];
        if (i >= 1 && word == "儿" && !out.empty()
            && !find_in_set(kXEng, out.back().second)) {
            out.back().first += word;
        } else {
            out.emplace_back(word, pos);
        }
    }
    return out;
}

// -------------------- public API --------------------

WordPosVec ToneSandhi::PreMergeForModify(
    const WordPosVec& seg) const {
    auto a = merge_bu(seg);
    a = merge_yi(a);
    a = merge_reduplication(a);
    a = merge_continuous_three_tones(a);
    a = merge_continuous_three_tones_2(a);
    a = merge_er(a);
    return a;
}

std::vector<std::string> ToneSandhi::ModifiedTone(
    const std::string& word, const std::string& pos,
    std::vector<std::string> finals) const {
    finals = bu_sandhi(word, std::move(finals));
    finals = yi_sandhi(word, std::move(finals));
    finals = neural_sandhi(word, pos, std::move(finals));
    finals = three_sandhi(word, std::move(finals));
    return finals;
}

} // namespace rs::kokoro_zh
