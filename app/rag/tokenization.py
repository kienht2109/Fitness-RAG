import tiktoken


TOKEN_ENCODING = "cl100k_base"


class TokenCounter:
    def __init__(self, encoding_name: str = TOKEN_ENCODING) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def split(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        step = max_tokens - overlap_tokens
        return [
            self._encoding.decode(tokens[start : start + max_tokens]).strip()
            for start in range(0, len(tokens), step)
            if tokens[start : start + max_tokens]
        ]
