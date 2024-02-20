# Typo Correction with Language Models

## Project Description

This project focuses on the development of a typo correction system using large language models. Given a sentence with typographical errors, our objective is to identify and replace the incorrect words with their most plausible corrections. The solution leverages the powerful contextual understanding of pre-trained language models to predict suitable replacements for masked tokens that represent typos in sentences.

### Background

Typographical errors are common in text data, especially with the increasing use of digital communication platforms. Correcting these errors is essential for improving the readability and understanding of the text. Traditional spell-checkers often rely on dictionary lookups and edit distances, which may not always capture the contextual appropriateness of a word in a sentence. This project aims to address these limitations by employing a context-aware approach using language models.

### The Challenge

The task involves processing sentences with identified typo positions, masking these typos, and utilizing a language model to suggest the most contextually fitting replacements. This approach benefits from the language model's training on extensive English language data, enabling it to grasp the semantic nuances required to fill in the masked positions accurately.

### Implementation Details

- **Input Format**: The input consists of sentences with identified typos, formatted as a comma-separated list of typo indices followed by the sentence itself.
  
- **Language Model**: We use `distilbert-base-uncased` from the Hugging Face's Transformers library for generating replacement suggestions. This model has been chosen for its balance between performance and computational efficiency.
