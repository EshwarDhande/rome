import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied directly to the model.
    """

    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text"):
        # Initialization method for the TokenizedDataset class.
        # It takes a text_dataset, tokenizer, maximum length (maxlen), and field as parameters.
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        # If the text_dataset has an "info" attribute, it is assigned to self.info.
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        # Returns the length of the text_dataset.
        return len(self.text_dataset)

    def __getitem__(self, i):
        # Retrieves the i-th item from the text_dataset.
        text = self.text_dataset[i]
        
        # If a specific field is specified, extract the text from that field.
        if self.field is not None:
            text = text[self.field]

        # Use the tokenizer to encode the text into token_list.
        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )

        # Generate position_ids as a list of integers ranging from 0 to len(token_list).
        position_ids = list(range(len(token_list)))

        # Create an attention_mask as a list of ones with the same length as token_list.
        attention_mask = [1] * len(token_list)

        # Return a dictionary containing input_ids, position_ids, and attention_mask.
        return dict(
            input_ids=torch.tensor(token_list),
            position_ids=torch.tensor(position_ids),
            attention_mask=torch.tensor(attention_mask),
        )


def dict_to_(data, device):
    """
    Moves a dictionary of tensors to the specified device.
    """
    for k in data:
        data[k] = data[k].to(device)
    return data


def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))       #Sorts the items in descending order of the length of the input_ids.
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}     #If the maximum length is 0, return a dictionary containing tensors of zeros.
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)     #Pads the sequences in the batch so that they are all the same length as the longest.
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
