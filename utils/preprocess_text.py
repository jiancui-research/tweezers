import re
from textacy import preprocessing
from defang import refang

# preprocessing rules

# Whitespace
_RE_WS = re.compile(r'\s+')

# Words containing non-alphanumeric characters
UNCOMMON_CHARS = r'[^\x00-\x7F\x80-\xFF]'
_RE_UNCOMMON = re.compile(UNCOMMON_CHARS)

_RE_BTC_ADDR = re.compile(r'([13]|bc1)[A-HJ-NP-Za-km-z1-9]{27,34}')
_RE_ETH_ADDR = re.compile(r'0x[a-fA-F0-9]{40}')
_RE_LTC_ADDR = re.compile(r'(ltc1|[LM])[a-zA-HJ-NP-Z0-9]{26,40}')

_RE_LONGWORD = re.compile(r'(\b|\B)\S{38,}(\b|\B)')

# IP address regex
IPV4SEG = r'(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
IPV4ADDR = r'(?:(?:' + IPV4SEG + r'\.){3,3}' + IPV4SEG + r')'
IPV6SEG = r'(?:(?:[0-9a-fA-F]){1,4})'
IPV6GROUPS = (
    r'(?:' + IPV6SEG + r':){7,7}' + IPV6SEG,                  # 1:2:3:4:5:6:7:8
    # 1::                                 1:2:3:4:5:6:7::
    r'(?:' + IPV6SEG + r':){1,7}:',
    # 1::8               1:2:3:4:5:6::8   1:2:3:4:5:6::8
    r'(?:' + IPV6SEG + r':){1,6}:' + IPV6SEG,
    # 1::7:8             1:2:3:4:5::7:8   1:2:3:4:5::8
    r'(?:' + IPV6SEG + r':){1,5}(?::' + IPV6SEG + r'){1,2}',
    # 1::6:7:8           1:2:3:4::6:7:8   1:2:3:4::8
    r'(?:' + IPV6SEG + r':){1,4}(?::' + IPV6SEG + r'){1,3}',
    # 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8
    r'(?:' + IPV6SEG + r':){1,3}(?::' + IPV6SEG + r'){1,4}',
    # 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8
    r'(?:' + IPV6SEG + r':){1,2}(?::' + IPV6SEG + r'){1,5}',
    # 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8
    IPV6SEG + r':(?:(?::' + IPV6SEG + r'){1,6})',
    # ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::
    r':(?:(?::' + IPV6SEG + r'){1,7}|:)',
    # fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)
    r'fe80:(?::' + IPV6SEG + r'){0,4}%[0-9a-zA-Z]{1,}',
    # ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
    r'::(?:ffff(?::0{1,4}){0,1}:){0,1}[^\s:]' + IPV4ADDR,
    r'(?:' + IPV6SEG + r':){1,6}:?[^\s:]' + IPV4ADDR
)
_RE_IPV4 = re.compile(IPV4ADDR)
_RE_IPV6 = re.compile('|'.join(['(?:{})'.format(g) for g in IPV6GROUPS[::-1]]))

_RE_URL_ONION = re.compile(r'(?:https?://)?(?:www)?(\S*?\.onion)(\S*)?')

# hashtag
_RE_HASHTAG = re.compile(r'\B#([A-Za-z_][A-Za-z0-9_]*)')
_RE_ENDHASHTAG = re.compile(r'\B#([A-Za-z_][A-Za-z0-9_]*)$')
_RE_MENTION = re.compile(r'\B@([A-Za-z_][A-Za-z0-9_]*)')

_RE_CVE = re.compile(r'CVE-\d{4}-\d{4,7}')

_RE_MD5 = re.compile('(?:^|[^A-Fa-f0-9])(?P<hash>[A-Fa-f0-9]{32})(?:$|[^A-Fa-f0-9])')
_RE_SHA1 = re.compile('(?:^|[^A-Fa-f0-9])(?P<hash>[A-Fa-f0-9]{32})(?:$|[^A-Fa-f0-9])')

def clean_hashtag(text):
    while 1:
        m = re.search(_RE_HASHTAG, text)
        if m == None:
            break
        text = text[:m.start()] + text[m.start()+1:]

    return text


def clean_mention(text):
    while 1:
        m = re.search(_RE_MENTION, text)
        if m == None:
            break
        text = text[:m.start()] + text[m.start()+1:]

    return text

def remove_endhashtag(text):
    while re.findall(_RE_ENDHASHTAG, text):
        text = _RE_ENDHASHTAG.sub('', text)
        text = text.strip()
        
    return text

def remove_hashtag(text):
    return _RE_HASHTAG.sub('', text)


def remove_mention(text):
    return _RE_MENTION.sub('', text)

def preprocess_text_for_keybert(text):
    
    # remove non-alphanuemeric characters
    text = re.sub(_RE_UNCOMMON, '', text)

    # refang text
    text = refang(text)

    text = clean_hashtag(text)
    text = clean_mention(text)

    text = _RE_WS.sub(' ', text).strip()

    return text

# preprocess text
def preprocess_text(text, lm):

    # remove non-alphanuemeric characters
    _text = re.sub(_RE_UNCOMMON, '', text)

    # refang text
    _text = refang(_text)

    # _text = remove_endhashtag(_text)
    # clean hashtag, mention: remove # if there is a hashtag, mention
    if not lm=='bertweet':
        _text = clean_hashtag(_text)
        _text = clean_mention(_text)
    # else:
        # print("bertweet not cleaning hash and user mention")
    
    _text = re.sub('http[s]?://t.co/\S+', 'ID_TWITTER_URL', _text)

    # replace emails and phone numbers
    _text = preprocessing.replace.emails(_text, repl="ID_EMAIL")
    _text = preprocessing.replace.phone_numbers(_text, repl="ID_PHONE")

    # replace IP addresses
    _text = _RE_IPV4.sub('ID_IP_ADDRESS', _text)
    _text = _RE_IPV6.sub('ID_IP_ADDRESS', _text)

    # replace hash 
    _text = _RE_MD5.sub('ID_MD5', _text)
    _text = _RE_SHA1.sub('ID_SHA1', _text)

    # replace CVE
    _text = _RE_CVE.sub('ID_CVE', _text)

    # replace URLs
    _text = _RE_URL_ONION.sub('ID_ONION_URL', _text)

    _text = re.sub('http[s]?://\S+', 'ID_NORMAL_URL', _text)

    # replace crypto addresses
    _text = _RE_BTC_ADDR.sub('ID_BTC_ADDRESS', _text)
    _text = _RE_ETH_ADDR.sub('ID_ETH_ADDRESS', _text)
    _text = _RE_LTC_ADDR.sub('ID_LTC_ADDRESS', _text)

    # for each word in _text, if the word length is greater than 38, then replace it with ID_LONG_WORD
    _text = _RE_LONGWORD.sub('ID_LONG_WORD', _text)

    # whitespace stripping
    text_preprocessed = _RE_WS.sub(' ', _text).strip()

    return text_preprocessed

# preprocess text


def preprocess_text_rm(text):

    # remove non-alphanuemeric characters
    _text = re.sub(_RE_UNCOMMON, '', text)

    # refang text
    _text = refang(_text)

    # clean hashtag, mention: remove # if there is a hashtag, mention
    _text = clean_hashtag(_text)
    _text = clean_mention(_text)

    # remove emails and phone numbers
    _text = preprocessing.replace.emails(_text, repl="")
    _text = preprocessing.replace.phone_numbers(_text, repl="")

    # remove IP addresses
    _text = _RE_IPV4.sub('', _text)
    _text = _RE_IPV6.sub('', _text)

    # remove URLs
    _text = _RE_URL_ONION.sub('', _text)
    _text = re.sub('http[s]?://\S+', '', _text)

    # remove crypto addresses
    _text = _RE_BTC_ADDR.sub('', _text)
    _text = _RE_ETH_ADDR.sub('', _text)
    _text = _RE_LTC_ADDR.sub('', _text)

    # for each word in _text, if the word length is greater than 38, then remove it with ID_LONG_WORD
    _text = _RE_LONGWORD.sub('ID_LONG_WORD', _text)

    # whitespace stripping
    text_preprocessed = _RE_WS.sub(' ', _text).strip()

    return text_preprocessed
