/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief Minimal, dependency-free JSON reader for the rocalution_opt tool.
 *
 *  rocALUTION (and rocsparse / hipsparse) intentionally avoid third-party JSON
 *  libraries, so this is a small hand-rolled recursive-descent parser for JSON
 *  (objects, arrays, strings, numbers, booleans, null).
 *
 *  As a convenience for hand-written config files, the parser also accepts JSONC-style
 *  comments (`//` to end of line and `/* ... *​/` blocks); these are treated as
 *  whitespace. This lets the example configs stay self-documenting while remaining a
 *  strict-JSON superset.
 */

#ifndef ROCALUTION_OPT_JSON_HPP_
#define ROCALUTION_OPT_JSON_HPP_

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocalution_opt
{
    /*! \brief A parsed JSON value (tagged union). */
    class JsonValue
    {
    public:
        enum class Type
        {
            Null,
            Bool,
            Number,
            String,
            Array,
            Object
        };

        Type type = Type::Null;

        bool                                 boolean = false;
        double                               number  = 0.0;
        std::string                          str;
        std::vector<JsonValue>               array;
        std::map<std::string, JsonValue>     object;

        bool is_null() const
        {
            return type == Type::Null;
        }
        bool is_bool() const
        {
            return type == Type::Bool;
        }
        bool is_number() const
        {
            return type == Type::Number;
        }
        bool is_string() const
        {
            return type == Type::String;
        }
        bool is_array() const
        {
            return type == Type::Array;
        }
        bool is_object() const
        {
            return type == Type::Object;
        }

        /*! \brief Return the object member with the given key, or nullptr if absent. */
        const JsonValue* find(const std::string& key) const
        {
            if(type != Type::Object)
            {
                return nullptr;
            }
            auto it = object.find(key);
            return it == object.end() ? nullptr : &it->second;
        }
    };

    /*! \brief Recursive-descent JSON parser. Throws std::runtime_error on malformed input. */
    class JsonParser
    {
    public:
        explicit JsonParser(std::string text)
            : text_(std::move(text))
        {
        }

        JsonValue parse()
        {
            skip_ws();
            JsonValue v = parse_value();
            skip_ws();
            if(pos_ != text_.size())
            {
                fail("trailing characters after top-level JSON value");
            }
            return v;
        }

    private:
        std::string text_;
        size_t      pos_ = 0;

        [[noreturn]] void fail(const std::string& msg) const
        {
            // Compute line/column for a helpful error message.
            size_t line = 1;
            size_t col  = 1;
            for(size_t i = 0; i < pos_ && i < text_.size(); ++i)
            {
                if(text_[i] == '\n')
                {
                    ++line;
                    col = 1;
                }
                else
                {
                    ++col;
                }
            }
            throw std::runtime_error("JSON parse error at line " + std::to_string(line) + ", column "
                                     + std::to_string(col) + ": " + msg);
        }

        char peek() const
        {
            return pos_ < text_.size() ? text_[pos_] : '\0';
        }

        char get()
        {
            return pos_ < text_.size() ? text_[pos_++] : '\0';
        }

        void skip_ws()
        {
            while(pos_ < text_.size())
            {
                char c = text_[pos_];
                if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
                {
                    ++pos_;
                }
                else if(c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/')
                {
                    // Line comment: skip to end of line.
                    pos_ += 2;
                    while(pos_ < text_.size() && text_[pos_] != '\n')
                    {
                        ++pos_;
                    }
                }
                else if(c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '*')
                {
                    // Block comment: skip to the closing '*/'.
                    pos_ += 2;
                    while(pos_ + 1 < text_.size()
                          && !(text_[pos_] == '*' && text_[pos_ + 1] == '/'))
                    {
                        ++pos_;
                    }
                    if(pos_ + 1 >= text_.size())
                    {
                        fail("unterminated block comment");
                    }
                    pos_ += 2; // consume closing '*/'
                }
                else
                {
                    break;
                }
            }
        }

        JsonValue parse_value()
        {
            skip_ws();
            char c = peek();
            switch(c)
            {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return parse_string();
            case 't':
            case 'f':
                return parse_bool();
            case 'n':
                return parse_null();
            default:
                if(c == '-' || (c >= '0' && c <= '9'))
                {
                    return parse_number();
                }
                fail(std::string("unexpected character '") + c + "'");
            }
        }

        JsonValue parse_object()
        {
            JsonValue v;
            v.type = JsonValue::Type::Object;
            get(); // consume '{'
            skip_ws();
            if(peek() == '}')
            {
                get();
                return v;
            }
            while(true)
            {
                skip_ws();
                if(peek() != '"')
                {
                    fail("expected string key in object");
                }
                std::string key = parse_string().str;
                skip_ws();
                if(get() != ':')
                {
                    fail("expected ':' after object key");
                }
                v.object[key] = parse_value();
                skip_ws();
                char n = get();
                if(n == ',')
                {
                    continue;
                }
                if(n == '}')
                {
                    break;
                }
                fail("expected ',' or '}' in object");
            }
            return v;
        }

        JsonValue parse_array()
        {
            JsonValue v;
            v.type = JsonValue::Type::Array;
            get(); // consume '['
            skip_ws();
            if(peek() == ']')
            {
                get();
                return v;
            }
            while(true)
            {
                v.array.push_back(parse_value());
                skip_ws();
                char n = get();
                if(n == ',')
                {
                    continue;
                }
                if(n == ']')
                {
                    break;
                }
                fail("expected ',' or ']' in array");
            }
            return v;
        }

        JsonValue parse_string()
        {
            JsonValue v;
            v.type = JsonValue::Type::String;
            get(); // consume opening quote
            std::string s;
            while(true)
            {
                if(pos_ >= text_.size())
                {
                    fail("unterminated string");
                }
                char c = get();
                if(c == '"')
                {
                    break;
                }
                if(c == '\\')
                {
                    char e = get();
                    switch(e)
                    {
                    case '"':
                        s.push_back('"');
                        break;
                    case '\\':
                        s.push_back('\\');
                        break;
                    case '/':
                        s.push_back('/');
                        break;
                    case 'b':
                        s.push_back('\b');
                        break;
                    case 'f':
                        s.push_back('\f');
                        break;
                    case 'n':
                        s.push_back('\n');
                        break;
                    case 'r':
                        s.push_back('\r');
                        break;
                    case 't':
                        s.push_back('\t');
                        break;
                    default:
                        fail("unsupported escape sequence in string");
                    }
                }
                else
                {
                    s.push_back(c);
                }
            }
            v.str = s;
            return v;
        }

        JsonValue parse_number()
        {
            size_t start = pos_;
            if(peek() == '-')
            {
                get();
            }
            while(true)
            {
                char c = peek();
                if((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+'
                   || c == '-')
                {
                    get();
                }
                else
                {
                    break;
                }
            }
            std::string num = text_.substr(start, pos_ - start);
            JsonValue   v;
            v.type   = JsonValue::Type::Number;
            v.number = std::strtod(num.c_str(), nullptr);
            return v;
        }

        JsonValue parse_bool()
        {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            if(text_.compare(pos_, 4, "true") == 0)
            {
                v.boolean = true;
                pos_ += 4;
            }
            else if(text_.compare(pos_, 5, "false") == 0)
            {
                v.boolean = false;
                pos_ += 5;
            }
            else
            {
                fail("invalid literal (expected true/false)");
            }
            return v;
        }

        JsonValue parse_null()
        {
            if(text_.compare(pos_, 4, "null") == 0)
            {
                pos_ += 4;
                JsonValue v;
                v.type = JsonValue::Type::Null;
                return v;
            }
            fail("invalid literal (expected null)");
        }
    };

} // namespace rocalution_opt

#endif // ROCALUTION_OPT_JSON_HPP_
