import re

_BRACKET_RE = re.compile(r'\[([^\[\]]+)\]')
_PROCESS_PID_RE = re.compile(r'(\b[a-zA-Z][\w._-]+)\[\d+\]')
_NUMERIC_RE = re.compile(r'^\d+$')
_TIMESTAMP_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}|^\d{2}:\d{2}:\d{2}|^\w{3}\s+\d{1,2}'
)


def extract_bracket_tokens(log_line):
    tokens = []

    # Extract content inside brackets: [sssd], [sbus_server]
    for tok in _BRACKET_RE.findall(log_line):
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if _NUMERIC_RE.match(tok):
            continue
        if _TIMESTAMP_RE.match(tok):
            continue
        tokens.append(tok.lower())

    # Extract process name before [PID]: krb5kdc[21363] -> krb5kdc
    for tok in _PROCESS_PID_RE.findall(log_line):
        tok = tok.strip().lower()
        if len(tok) >= 2 and tok not in tokens:
            tokens.append(tok)

    return tokens


def extract_all_bracket_tokens(df):
    all_tokens = set()
    per_row_tokens = []
    for text in df["text"]:
        tokens = extract_bracket_tokens(str(text))
        per_row_tokens.append(tokens)
        all_tokens.update(tokens)
    return list(all_tokens), per_row_tokens
