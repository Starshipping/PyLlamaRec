# Prompt templates

This directory contains template styles for the prompts used to finetune LoRA models.

## Format

A template is described via a JSON file with the following keys:

- `prompt_input`: The template to use when input is not None. Uses `{instruction}` and `{input}` placeholders.
- `prompt_no_input`: The template to use when input is None. Uses `{instruction}` placeholders.
- `description`: A short description of the template, with possible use cases.
- `response_split`: The text to use as separator when cutting real response from the model output.

No `{response}` placeholder was used, since the respons