from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, apply_rlvr=None) -> list:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = [apply_chat_template(chat, tokenize=False, add_generation_prompt=True)]
    elif apply_rlvr:
        prompt = [data[input_key]]
        if input_template:
            prompt = [input_template.format(prompt[0])]
        prompt.append(data["label"])
    else:
        prompt = [data[input_key]]
        if input_template:
            prompt = [input_template.format(prompt[0])]
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        apply_rlvr = getattr(self.strategy.args, "apply_rlvr", False)


        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, apply_rlvr)
            if apply_rlvr:
                self.prompts.append(prompt)
            else:
                self.prompts.append(prompt[0])

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
