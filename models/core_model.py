import torch
import torch.nn as nn
import fairscale.nn as fairnn
from typing import List, Optional, Union

class NextGenTorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize an expanded vocabulary for tokenization and decoding
        self.vocab = {}
        # Initialize with byte-level tokens
        for i in range(256):
            self.vocab[bytes([i]).decode('latin-1')] = i

        # Add special tokens
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[BOS]', '[EOS]']
        for i, token in enumerate(special_tokens):
            self.vocab[token] = 256 + i

        # Add common words and subwords
        common_tokens = ['the', 'of', 'and', 'to', 'in', 'a', 'is', 'that', 'for', 'it',
                         'with', 'as', 'was', 'on', 'be', 'at', 'by', 'this', 'have', 'from',
                         'or', 'one', 'had', 'not', 'but', 'what', 'all', 'were', 'we', 'when',
                         'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which', 'she', 'do',
                         'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then',
                         'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him', 'into',
                         'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see', 'number', 'no',
                         'way', 'could', 'people', 'my', 'than', 'first', 'water', 'been', 'call', 'who',
                         'oil', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come']
        start_index = len(self.vocab)
        for i, token in enumerate(common_tokens):
            self.vocab[token] = start_index + i

        # Add common subwords
        common_subwords = ['ing', 'ed', 'er', 'es', 'tion', 'ly', 'ment', 'ness', 'ous', 'ity',
                           'able', 'ible', 'al', 'ive', 'ful', 'ic', 'ism', 'ist', 'ize', 'less']
        start_index = len(self.vocab)
        for i, subword in enumerate(common_subwords):
            self.vocab[subword] = start_index + i

        # Initialize BPE merges
        self.bpe_merges = {}

        # Function to get character pair statistics
        def get_stats(self, word_freqs):
            pairs = {}
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            return pairs

        # Function to merge word frequencies
        def merge_word_freqs(self, pair, word_freqs):
            new_word_freqs = {}
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            return new_word_freqs

        # Function to update BPE merges
        def update_bpe_merges(self, text, num_merges=1000):
            word_freqs = {}
            for word in text.split():
                word = ' '.join(list(word)) + ' </w>'  # Add end-of-word token
                if word not in word_freqs:
                    word_freqs[word] = 1
                else:
                    word_freqs[word] += 1

            for i in range(num_merges):
                pairs = self.get_stats(word_freqs)
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                self.bpe_merges.append(best)
                word_freqs = self.merge_word_freqs(best, word_freqs)

        self.get_stats = get_stats
        self.merge_word_freqs = merge_word_freqs
        self.update_bpe_merges = update_bpe_merges

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Initialize placeholder layers
        self.padding_idx = self.vocab['[PAD]']
        self.vocab_size = self.get_vocab_size()
        self.embedding = nn.Embedding(self.vocab_size, self.config['hidden_size'], padding_idx=self.padding_idx)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.config['hidden_size'], nhead=self.config['num_attention_heads'], batch_first=True),
            num_layers=self.config['num_layers']
        )
        self.output_layer = nn.Linear(self.config['hidden_size'], self.vocab_size)

    def get_vocab_size(self):
        return len(self.vocab)  # No need to add 1, as [PAD] is already in the vocab

    def forward(self, input_ids, attention_mask=None):
        try:
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)

            print(f"Input IDs shape: {input_ids.shape}, Device: {device}")
            if input_ids.dim() != 2:
                raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}")

            batch_size, seq_length = input_ids.shape
            print(f"Batch size: {batch_size}, Sequence length: {seq_length}")

            # Check if input_ids are within the valid range
            if torch.max(input_ids) >= len(self.vocab):
                raise ValueError(f"Input contains token IDs outside of vocabulary range. Max token ID: {torch.max(input_ids).item()}, Vocab size: {len(self.vocab)}")

            embedded = self.embedding(input_ids)
            print(f"Embedded shape: {embedded.shape}")

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                print(f"Original attention mask shape: {attention_mask.shape}")
                if attention_mask.shape != input_ids.shape:
                    raise ValueError(f"Attention mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")

                # Ensure attention mask is 2D
                if attention_mask.dim() != 2:
                    attention_mask = attention_mask.squeeze()
                    if attention_mask.dim() != 2:
                        raise ValueError(f"Attention mask must be 2D, got shape {attention_mask.shape}")

                # Invert the attention mask for PyTorch transformer
                attention_mask = (1.0 - attention_mask) * -10000.0
                attention_mask = attention_mask.masked_fill(attention_mask == -10000, float('-inf'))
                print(f"Processed attention mask shape: {attention_mask.shape}")
            else:
                print("No attention mask provided")
                attention_mask = None

            print(f"Transformer input shape: {embedded.shape}")
            transformer_output = self.transformer(embedded, src_key_padding_mask=attention_mask)
            print(f"Transformer output shape: {transformer_output.shape}")

            optimized_output = self.apply_phi3_optimizations(transformer_output)
            print(f"Optimized output shape: {optimized_output.shape}")

            extended_output = self.handle_extended_context(optimized_output)
            print(f"Extended output shape: {extended_output.shape}")

            logits = self.output_layer(extended_output)
            print(f"Logits shape: {logits.shape}")

            expected_shape = (batch_size, seq_length, self.vocab_size)
            if logits.shape != expected_shape:
                raise ValueError(f"Logits shape mismatch. Expected {expected_shape}, got {logits.shape}")

            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite values detected in logits")

            return logits

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"Attention mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
            print(f"Device: {device}")
            print(f"Model parameters: {sum(p.numel() for p in self.parameters())}")
            print(f"Vocab size: {len(self.vocab)}")
            print(f"Embedding layer shape: {self.embedding.weight.shape}")
            print(f"Output layer shape: {self.output_layer.weight.shape}")
            print(f"Transformer layers: {self.transformer.num_layers}")
            print(f"Hidden size: {self.config['hidden_size']}")
            print(f"Attention heads: {self.config['num_attention_heads']}")
            raise

    def train_step(self, batch):
        input_ids, attention_mask, labels = batch

        # Forward pass
        outputs = self(input_ids, attention_mask)

        # Calculate loss
        loss = self.calculate_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()

        return loss.item()

    def evaluate(self, dataset):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataset:
                input_ids, attention_mask, labels = batch
                outputs = self(input_ids, attention_mask)
                loss = self.calculate_loss(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(dataset)

    def apply_phi3_optimizations(self, outputs):
        # Implement Phi3-inspired optimizations
        # This is a placeholder and should be replaced with actual optimizations
        return outputs

    def handle_extended_context(self, outputs):
        # Implement Grok-inspired extended context length handling
        # This is a placeholder and should be replaced with actual implementation
        return outputs

    def calculate_loss(self, outputs, labels):
        # Implement loss calculation
        # This is a placeholder and should be replaced with actual loss calculation
        return torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

    # Custom methods for text generation
    def generate(self, prompt: str, max_length: int = 100, stop: List[str] = None, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, num_beams: int = 4, max_retries: int = 3, max_iterations: int = 1000) -> str:
        try:
            # Enhanced input validation
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("max_length must be a positive integer")
            if not isinstance(temperature, float) or temperature <= 0 or temperature > 2:
                raise ValueError("temperature must be a positive float between 0 and 2")
            if not isinstance(top_p, float) or top_p <= 0 or top_p > 1:
                raise ValueError("top_p must be a float between 0 and 1")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")
            if not isinstance(num_beams, int) or num_beams <= 0:
                raise ValueError("num_beams must be a positive integer")
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise ValueError("max_iterations must be a positive integer")

            input_ids = self.tokenize(prompt)
            if input_ids.numel() == 0:
                raise ValueError("Tokenization resulted in empty input")

            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            context_length = input_ids.size(-1)

            print(f"Generation started. Initial input_ids shape: {input_ids.shape}, Device: {device}, Context length: {context_length}")

            # Initialize beam search
            beam_scores = torch.zeros(num_beams, device=device)
            beam_inputs = input_ids.repeat(num_beams, 1)
            done_beams = []

            for i in range(min(max_length - context_length, max_iterations)):
                with torch.no_grad():
                    try:
                        outputs = self(beam_inputs)
                        if outputs is None or outputs.numel() == 0:
                            raise ValueError("Forward pass returned empty or None output")
                        if outputs.shape[1] != beam_inputs.shape[1]:
                            raise ValueError(f"Output shape mismatch. Expected {beam_inputs.shape[1]}, got {outputs.shape[1]}")

                        next_token_logits = outputs[:, -1, :].clone()

                        # Apply temperature
                        next_token_logits = next_token_logits / max(temperature, 1e-8)

                        # Apply top-k filtering
                        top_k = min(top_k, next_token_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                        next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = float('-inf')

                        # Apply top-p (nucleus) filtering
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                        # Calculate probabilities
                        probs = torch.softmax(next_token_logits, dim=-1)

                        # Sample next tokens for each beam
                        next_tokens = torch.multinomial(probs, num_samples=2)  # Sample 2 tokens for potential backoff
                        beam_scores = beam_scores.unsqueeze(1) + next_token_logits.gather(1, next_tokens)

                        # Select top beams
                        beam_scores, beam_indices = torch.topk(beam_scores.view(-1), num_beams)
                        beam_tokens = next_tokens.view(-1)[beam_indices]

                        # Update beam inputs
                        beam_inputs = torch.cat([beam_inputs[beam_indices // 2], beam_tokens.unsqueeze(1)], dim=-1)

                        # Check for completed beams
                        new_done_beams = []
                        for j, beam in enumerate(beam_inputs):
                            decoded_beam = self.decode(beam)
                            if self._check_stop_condition(decoded_beam, stop) or beam.size(0) >= max_length:
                                new_done_beams.append((beam_scores[j], beam))
                                beam_scores[j] = float('-inf')

                        # Add new completed beams and remove them from active beams
                        done_beams.extend(new_done_beams)
                        if new_done_beams:
                            active_beam_indices = beam_scores != float('-inf')
                            beam_inputs = beam_inputs[active_beam_indices]
                            beam_scores = beam_scores[active_beam_indices]
                            if len(beam_inputs) == 0:
                                break

                        print(f"Step {i+1}: Generated tokens, Shape: {beam_inputs.shape}")
                        print(f"Sample decoded text: {self.decode(beam_inputs[0])}")

                        # Early stopping if no progress is made
                        if i > 0 and all(beam.size(0) == beam_inputs[0].size(0) for beam in beam_inputs):
                            print("No progress made in beam search. Stopping early.")
                            break

                    except (RuntimeError, IndexError, ValueError) as e:
                        print(f"Step {i+1}: Error occurred: {str(e)}")
                        print(f"Current beam_inputs shape: {beam_inputs.shape}")
                        print(f"Current logits shape: {next_token_logits.shape if 'next_token_logits' in locals() else 'N/A'}")
                        print(f"Current device: {device}")
                        if i == 0:
                            raise
                        else:
                            print(f"Attempting to continue generation...")
                            continue

                if i == max_iterations - 1:
                    print(f"Reached maximum iterations ({max_iterations}). Stopping generation.")

            # Select the best beam
            if done_beams:
                best_beam = max(done_beams, key=lambda x: x[0])[1]
            elif beam_inputs.size(0) > 0:
                best_beam = beam_inputs[beam_scores.argmax()]
            else:
                raise ValueError("No valid beams generated")

            print(f"Generation completed. Final generated tokens length: {best_beam.size(0)}")

            # Decode the best beam to get the generated text
            generated_text = self.decode(best_beam)

            # Ensure the generated text starts with the prompt
            if not generated_text.startswith(prompt):
                generated_text = prompt + " " + generated_text

            # Remove any remaining [UNK] tokens
            generated_text = generated_text.replace("[UNK]", "")

            return generated_text.strip()

        except Exception as e:
            error_message = f"Error during generation: {str(e)}"
            print(error_message)
            print(f"Input shape: {input_ids.shape if 'input_ids' in locals() else 'N/A'}, "
                  f"Device: {device if 'device' in locals() else 'N/A'}, "
                  f"Context length: {context_length if 'context_length' in locals() else 'N/A'}")
            print(f"Model parameters: {sum(p.numel() for p in self.parameters())}")
            print(f"Vocab size: {len(self.vocab)}")
            raise  # Re-raise the exception instead of returning an error message

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        try:
            # Enhanced input validation and reshaping
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"Logits must be a torch.Tensor, got {type(logits)}")

            # Ensure logits are 2D and reshape if necessary
            original_shape = logits.shape
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() > 2:
                logits = logits.view(1, -1)

            vocab_size = len(self.vocab)
            if logits.size(1) != vocab_size:
                raise ValueError(f"Logits must have shape (1, {vocab_size}), got {logits.shape}")

            print(f"Vocab size: {vocab_size}, Original logits shape: {original_shape}, Reshaped logits shape: {logits.shape}")

            # Apply temperature with safeguard
            temperature = max(temperature, 1e-8)
            scaled_logits = logits / temperature

            # Apply top-k sampling with bounds checking
            top_k = min(max(1, top_k), vocab_size)
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)

            print(f"Top-k logits shape: {top_k_logits.shape}, Top-k indices shape: {top_k_indices.shape}")

            # Handle potential NaN or Inf values
            top_k_logits = torch.nan_to_num(top_k_logits, nan=-float('inf'), posinf=-float('inf'), neginf=-float('inf'))

            # Apply softmax with increased numerical stability
            top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(top_k_logits, dim=-1)

            print(f"Probabilities shape: {probs.shape}, Sum: {probs.sum().item():.4f}")

            # Handle zero or very small probabilities
            if (probs < 1e-8).all():
                print("Warning: All probabilities are very small. Using uniform distribution.")
                probs = torch.ones_like(probs) / probs.size(-1)

            # Sample the next token with error handling
            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                print(f"Error in multinomial sampling: {str(e)}. Using argmax instead.")
                next_token = probs.argmax(dim=-1, keepdim=True)

            # Map next_token back to the original token space
            sampled_token = top_k_indices[0, next_token[0]].unsqueeze(0)

            print(f"Sampled token: {sampled_token.item()}, Probability: {probs[0, next_token[0]].item():.4f}")

            # Ensure the sampled token is within the valid range
            if sampled_token.item() >= vocab_size:
                print(f"Warning: Sampled token {sampled_token.item()} is out of vocabulary range. Using UNK token.")
                return torch.tensor([self.vocab.get('[UNK]', vocab_size - 1)], device=logits.device)

            return sampled_token

        except Exception as e:
            print(f"Error in _sample_next_token: {str(e)}")
            print(f"Logits shape: {logits.shape}, Top-k: {top_k}, Temperature: {temperature}")
            print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")

            # Fallback to most probable token from original logits
            fallback_token = torch.argmax(logits, dim=-1)
            print(f"Falling back to most probable token: {fallback_token.item()}")

            # Ensure the fallback token is within the valid range
            if fallback_token.item() >= vocab_size:
                print(f"Warning: Fallback token {fallback_token.item()} is out of vocabulary range. Using UNK token.")
                return torch.tensor([self.vocab.get('[UNK]', vocab_size - 1)], device=logits.device)

            return fallback_token.unsqueeze(0)

    def _ensure_valid_token(self, token: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Helper method to ensure a valid token is returned."""
        if token.numel() == 0 or token[0].item() >= top_k_indices.size(1):
            print("Warning: Invalid token, using most probable token")
            return top_k_indices[0, 0].unsqueeze(0)
        return top_k_indices[0, token[0]].unsqueeze(0)

    def _check_stop_condition(self, generated_text: str, stop: List[str]) -> bool:
        if stop:
            for stop_sequence in stop:
                if stop_sequence in generated_text:
                    return True
        return False

    def batch_generate(self, prompts: List[str], max_length: int = 100, stop: List[str] = None) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, max_length, stop))
        return results

    def tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        try:
            if isinstance(text, str):
                text_to_tokenize = text
            elif isinstance(text, list):
                if not all(isinstance(item, str) for item in text):
                    raise ValueError("All items in the list must be strings")
                text_to_tokenize = " ".join(text)
            else:
                raise ValueError("Input must be a string or a list of strings")

            # Apply BPE tokenization
            tokens = self.bpe_tokenize(text_to_tokenize)

            unknown_token = self.vocab['[UNK]']
            token_ids = []
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(unknown_token)

            if not token_ids:
                raise ValueError("Tokenization resulted in empty token list")

            vocab_size = self.get_vocab_size()
            token_ids = [min(token_id, vocab_size - 1) for token_id in token_ids]

            tokenized = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            print(f"Tokenization successful. Shape: {tokenized.shape}, Device: {tokenized.device}")
            print(f"Unique token IDs: {torch.unique(tokenized).tolist()}")
            print(f"Max token ID: {tokenized.max().item()}, Vocab size: {vocab_size}")
            return tokenized
        except Exception as e:
            print(f"Error during tokenization: {str(e)}")
            print(f"Input text: {text[:100]}..." if len(text) > 100 else f"Input text: {text}")
            raise

    def bpe_tokenize(self, text: str) -> List[str]:
        def get_pairs(word):
            return set(zip(word[:-1], word[1:]))

        words = text.split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                word = list(word) + ['</w>']
                while True:
                    pairs = get_pairs(word)
                    if not pairs:
                        break
                    bigram = min(pairs, key=lambda pair: self.bpe_merges.get(pair, float('inf')))
                    if bigram not in self.bpe_merges:
                        break
                    first, second = bigram
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                            new_word.append(first + second)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word

                word = ''.join(word).replace('</w>', '')
                subwords = [subword for subword in word.split() if subword in self.vocab]
                if not subwords:
                    subwords = ['[UNK]']
                tokens.extend(subwords)
        return tokens

    def decode(self, outputs: torch.Tensor) -> str:
        try:
            if not isinstance(outputs, torch.Tensor):
                raise ValueError(f"Input must be a torch.Tensor, got {type(outputs)}")

            outputs = outputs.squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            # Convert to integers and handle float tensors
            if outputs.dtype in [torch.float32, torch.float64]:
                outputs = outputs.round().long()
            outputs = outputs.tolist()

            if isinstance(outputs, int):
                outputs = [outputs]
            elif isinstance(outputs, list) and any(isinstance(item, list) for item in outputs):
                outputs = [item for sublist in outputs for item in (sublist if isinstance(sublist, list) else [sublist])]

            decoded_tokens = []
            vocab_size = self.get_vocab_size()
            for token in outputs:
                try:
                    token = int(token)  # Ensure token is an integer
                    if token < 0 or token >= vocab_size:
                        print(f"Warning: Token {token} is out of vocabulary range. Using [UNK].")
                        decoded_tokens.append('[UNK]')
                    else:
                        subword = self.reverse_vocab.get(token)
                        if subword is None:
                            print(f"Warning: Token {token} not found in reverse vocabulary. Using [UNK].")
                            decoded_tokens.append('[UNK]')
                        else:
                            decoded_tokens.append(str(subword))  # Ensure subword is a string
                except ValueError:
                    print(f"Warning: Invalid token {token}. Using [UNK].")
                    decoded_tokens.append('[UNK]')

            # Merge subwords
            result = ' '.join(decoded_tokens).replace('▁', ' ').strip()
            print(f"Decoded text: {result}")
            return result

        except Exception as e:
            error_message = f"Error in decode method: {str(e)}"
            print(f"ERROR: {error_message}")
            return '[ERROR]'

    def get_vocab_size(self) -> int:
        return len(self.vocab)

def create_nextgentorch_model(model_size):
    # Function to create model based on size (1b, 2b, 7b, 16b, 32b, 64b, 128b)
    config = {
        "model_size": model_size,
        "num_layers": get_num_layers(model_size),
        "hidden_size": get_hidden_size(model_size),
        "num_attention_heads": get_num_attention_heads(model_size),
        "max_sequence_length": 8192,  # Extended context length inspired by Grok
    }
    return NextGenTorchModel(config)

def get_num_layers(model_size):
    # Define number of layers based on model size
    size_to_layers = {
        "1b": 24, "2b": 32, "7b": 48, "16b": 64,
        "32b": 80, "64b": 96, "128b": 112
    }
    return size_to_layers.get(model_size, 24)  # Default to 24 if size not found

def get_hidden_size(model_size):
    # Define hidden size based on model size
    size_to_hidden = {
        "1b": 1024, "2b": 1536, "7b": 2048, "16b": 3072,
        "32b": 4096, "64b": 5120, "128b": 6144
    }
    return size_to_hidden.get(model_size, 1024)  # Default to 1024 if size not found

def get_num_attention_heads(model_size):
    # Define number of attention heads based on model size
    size_to_heads = {
        "1b": 16, "2b": 24, "7b": 32, "16b": 48,
        "32b": 64, "64b": 80, "128b": 96
    }
    return size_to_heads.get(model_size, 16)  # Default to 16 if size not found
