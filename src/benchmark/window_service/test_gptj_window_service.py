import shutil
import tempfile

from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory
from .test_gpt2_window_service import TEST_TOKENS, TEST_PROMPT, TEST_TOKEN_IDS
from .test_utils import get_tokenizer_service


class TestGPTJWindowService:
    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.window_service = WindowServiceFactory.get_window_service("together/gpt-j-6b", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_sequence_length(self):
        assert self.window_service.max_sequence_length == 2048

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2049

    def test_tokenizer_name(self):
        assert self.window_service.tokenizer_name == "huggingface/gpt-j-6b"

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).tokens == TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(TEST_TOKEN_IDS) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == TEST_TOKENS

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, 2049 - 51)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(TEST_PROMPT, 2049 - 51 + 1)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 51 * 41 = 2091 tokens
        long_prompt: str = TEST_PROMPT * 41
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == 2049
        assert self.window_service.fits_within_context_window(truncated_long_prompt)

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 51
