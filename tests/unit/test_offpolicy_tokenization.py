"""Keep offline completion boundaries correct across tokenizer API versions."""

from ssd.kl_utils import tokenize_for_fkl
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


def test_student_and_privileged_teacher_share_completion_tokens():
    vocabulary = {
        word: i
        for i, word in enumerate(
            [
                "<unk>",
                "user",
                "question",
                "assistant",
                "yes",
                "<eos>",
                "system",
                "privileged",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% if message['role'] == 'assistant' %}<eos> {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    row = {
        "prompt": [{"role": "user", "content": "question"}],
        "teacher_prompt": [
            {"role": "system", "content": "privileged"},
            {"role": "user", "content": "question"},
        ],
        "answer": "yes",
    }
    result = tokenize_for_fkl(row, tokenizer, 32, 32)
    assert result is not None
    assert result["completion_start_student"] == 3
    assert result["completion_start_teacher"] == 5
    assert result["num_completion_tokens"] == 2
    assert (
        result["student_ids"][3:]
        == result["teacher_ids"][5:]
        == [vocabulary["yes"], vocabulary["<eos>"]]
    )
