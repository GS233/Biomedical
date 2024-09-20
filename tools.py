import torch
from torch.utils.data import Dataset, DataLoader
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer_1, tokenizer_2, tokenizer_3, max_length):
        self.data = dataframe
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['sentence'][idx]
        label_str = self.data['label'][idx]
        label = int(label_str)

        encoding_1 = self.tokenizer_1(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding_2 = self.tokenizer_2(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding_3 = self.tokenizer_3(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids_1': encoding_1['input_ids'].flatten(),
            'attention_mask_1': encoding_1['attention_mask'].flatten(),
            'input_ids_2': encoding_2['input_ids'].flatten(),
            'attention_mask_2': encoding_2['attention_mask'].flatten(),
            'input_ids_3': encoding_3['input_ids'].flatten(),
            'attention_mask_3': encoding_3['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer_1, tokenizer_2, tokenizer_3, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )