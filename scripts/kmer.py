import itertools
from typing import Any, Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast


class KmerTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        k: int = 3,
        overlap: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        self.k = k
        self.overlap = overlap

        # Store token values for later reference
        self.unk_token_value = unk_token
        self.sep_token_value = sep_token
        self.pad_token_value = pad_token
        self.cls_token_value = cls_token
        self.mask_token_value = mask_token

        # Initialize base tokenizer
        tokenizer = self._create_base_tokenizer(
            unk_token, sep_token, pad_token, cls_token, mask_token
        )

        # Initialize parent class
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # Build the complete vocabulary
        self._build_vocabulary()

        # Store token IDs for quick access
        self.unk_token_id = self._tokenizer.token_to_id(self.unk_token_value)
        self.sep_token_id = self._tokenizer.token_to_id(self.sep_token_value)
        self.pad_token_id = self._tokenizer.token_to_id(self.pad_token_value)
        self.cls_token_id = self._tokenizer.token_to_id(self.cls_token_value)
        self.mask_token_id = self._tokenizer.token_to_id(self.mask_token_value)

    def _create_base_tokenizer(
        self, unk_token, sep_token, pad_token, cls_token, mask_token
    ):
        # Create the base tokenizer with special tokens
        # Initialize with special tokens in the specific order requested
        special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
        vocab = {token: i for i, token in enumerate(special_tokens)}

        # Create base tokenizer with the initial vocabulary
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token))
        tokenizer.decoder = decoders.WordPiece()

        return tokenizer

    def _extract_kmers(self, sequence: str) -> List[str]:
        # Extract k-mers from a sequence
        sequence = sequence.upper()

        if len(sequence) < self.k:
            return [sequence]  # Return the whole sequence if shorter than k

        if self.overlap:
            # Overlapping k-mers (sliding window)
            return [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]
        else:
            # Non-overlapping k-mers
            kmers = [
                sequence[i : i + self.k]
                for i in range(0, len(sequence) - self.k + 1, self.k)
            ]
            # Add remaining part if not perfectly divisible by k
            remainder = len(sequence) % self.k
            if remainder > 0:
                kmers.append(sequence[-remainder:])
            return kmers

    def _build_vocabulary(self):
        # Build a complete k-mer vocabulary in alphabetical order
        # Start with the special tokens in the specified order
        special_tokens = [
            self.unk_token_value,
            self.sep_token_value,
            self.pad_token_value,
            self.cls_token_value,
            self.mask_token_value,
        ]
        vocab = {token: i for i, token in enumerate(special_tokens)}

        # Generate all possible k-mers for DNA (A, C, G, T)
        bases = ["A", "C", "G", "T"]

        # Generate all possible k-mers using itertools.product
        all_kmers = ["".join(kmer) for kmer in itertools.product(bases, repeat=self.k)]

        # Sort alphabetically and add to vocabulary
        all_kmers.sort()
        next_id = len(special_tokens)
        for kmer in all_kmers:
            vocab[kmer] = next_id
            next_id += 1

        # Update the tokenizer's model
        self._tokenizer.model = models.WordLevel(vocab, self.unk_token_value)

        return vocab

    def tokenize(self, text: str, **kwargs) -> List[str]:
        # Tokenize text into k-mers
        text = text.upper()
        kmers = self._extract_kmers(text)
        return kmers

    def encode(
        self, text: Union[str, List[str]], add_special_tokens: bool = True, **kwargs
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs using k-mer tokenization.

        Args:
            text: Text or list of texts to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            Token IDs
        """
        # For a single string input
        if isinstance(text, str):
            text = text.upper()
            kmers = self._extract_kmers(text)

            token_ids = []
            for kmer in kmers:
                token_id = self._tokenizer.token_to_id(kmer)
                if token_id is None:
                    token_id = self.unk_token_id
                token_ids.append(token_id)

            # Add special tokens if requested
            if add_special_tokens:
                token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]

            return token_ids

        # For a list of strings
        encoded_sequences = []
        for sequence in text:
            encoded = self.encode(
                sequence, add_special_tokens=add_special_tokens, **kwargs
            )
            encoded_sequences.append(encoded)

        return encoded_sequences

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, None] = None,
        truncation: Union[bool, str, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main tokenization method that handles padding, truncation, and other options.
        This method is called when the tokenizer is directly invoked as a function.
        """
        # First, encode the input text(s)
        is_batched = isinstance(text, (list, tuple))

        if not is_batched:
            text = [text]

        encoded_inputs = []
        for txt in text:
            # Encode each text
            token_ids = self.encode(txt, add_special_tokens=add_special_tokens)
            encoded_input = {"input_ids": token_ids}

            # Add attention mask (1 for real tokens, 0 for padding)
            if return_attention_mask:
                encoded_input["attention_mask"] = [1] * len(token_ids)

            # Add token_type_ids (all 0 for single sequence)
            if return_token_type_ids is not False:
                encoded_input["token_type_ids"] = [0] * len(token_ids)

            encoded_inputs.append(encoded_input)

        # Handle padding and truncation
        if padding == "max_length" or padding == True or truncation == True:
            # Determine the target length
            target_length = (
                max_length
                if max_length is not None
                else max(len(enc["input_ids"]) for enc in encoded_inputs)
            )

            for i, encoded_input in enumerate(encoded_inputs):
                # Truncate if necessary
                if truncation and len(encoded_input["input_ids"]) > target_length:
                    encoded_input["input_ids"] = encoded_input["input_ids"][
                        :target_length
                    ]
                    if "attention_mask" in encoded_input:
                        encoded_input["attention_mask"] = encoded_input[
                            "attention_mask"
                        ][:target_length]
                    if "token_type_ids" in encoded_input:
                        encoded_input["token_type_ids"] = encoded_input[
                            "token_type_ids"
                        ][:target_length]

                # Pad if necessary
                if padding and len(encoded_input["input_ids"]) < target_length:
                    padding_length = target_length - len(encoded_input["input_ids"])

                    # Pad input_ids with pad_token_id
                    encoded_input["input_ids"] = (
                        encoded_input["input_ids"]
                        + [self.pad_token_id] * padding_length
                    )

                    # Pad attention_mask with 0s
                    if "attention_mask" in encoded_input:
                        encoded_input["attention_mask"] = (
                            encoded_input["attention_mask"] + [0] * padding_length
                        )

                    # Pad token_type_ids with 0s
                    if "token_type_ids" in encoded_input:
                        encoded_input["token_type_ids"] = (
                            encoded_input["token_type_ids"] + [0] * padding_length
                        )

        # Combine results for batched input
        if not is_batched:
            # If input was a single text, return a single result
            result = encoded_inputs[0]
        else:
            # If input was a batch, return a batch of results
            result = {
                key: [encoded_input[key] for encoded_input in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        return result

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        # Handle different input types
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if not isinstance(token_ids[0], list):
            # Single sequence of token IDs

            # Filter out special tokens if requested
            if skip_special_tokens:
                special_token_ids = [
                    self.unk_token_id,
                    self.sep_token_id,
                    self.pad_token_id,
                    self.cls_token_id,
                    self.mask_token_id,
                ]
                token_ids = [id for id in token_ids if id not in special_token_ids]

            # Convert tokens back to k-mers
            kmers = []
            for token_id in token_ids:
                kmer = self._tokenizer.id_to_token(token_id)
                if kmer is not None:
                    kmers.append(kmer)

            # Reconstruct the sequence
            if self.overlap and len(kmers) > 1:
                # For overlapping k-mers, reconstruct by taking the first k-mer and
                # adding the last character of each subsequent k-mer
                sequence = kmers[0]
                for kmer in kmers[1:]:
                    if len(kmer) > 0:
                        sequence += kmer[-1]
                return sequence
            else:
                # For non-overlapping k-mers, just concatenate
                return "".join(kmers)
        else:
            # List of token ID sequences
            return [
                self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)
                for ids in token_ids
            ]
