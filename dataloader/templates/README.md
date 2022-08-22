# Prompt templates

This directory contains template styles for the prompts used to finetune LoRA models.

## Format

A template is described via a JSON file with the following keys:

- `prompt_input`: The template to use when input is not None. Uses `{instruction}` and `{input}` placeholders.
