# -*- coding: utf-8 -

import logging
import torch

from simpletransformers.language_generation import LanguageGenerationModel

"""Generate a sequence by using the prompts."""

use_cuda = torch.cuda.is_available()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def generate_text(prompt):
    model = LanguageGenerationModel("gpt2", "gpt2", args={"length": 256}, use_cuda=use_cuda)
    generated_text = model.generate(prompt, verbose=False)
    generated_text = '.'.join(generated_text[0].split('.')[:-1]) + '.'
    return generated_text


if __name__ == '__main__':
    prompt_text = ''
    print(generate_text(prompt_text))
