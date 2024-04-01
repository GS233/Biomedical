import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#############################################定义 BERT 模型和 tokenizer##############################################
# # BioBERT
# bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# bio_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# # BlueBERT
# blue_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
# blue_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

# # PubMedBERT
# pubmed_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# pubmed_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# SciBERT
model_path_1 = './model_path/scibert_scivocab_uncased'
scibert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
scibert_model = AutoModel.from_pretrained(model_path_1)
model_path_2 = './model_path/biobert'
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path_2)
biobert_model = AutoModel.from_pretrained(model_path_2)
model_path_3 = './model_path/bluebert'
bluebert_tokenizer = AutoTokenizer.from_pretrained(model_path_3)
bluebert_model = AutoModel.from_pretrained(model_path_3)
print("model load")
# ############################################读取数据#################################################################
df_train = pd.read_csv('/root/yuanzhu/Biomedical/data/ddi2013ms/train.tsv', sep='\t')
df_dev = pd.read_csv('/root/yuanzhu/Biomedical/data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('/root/yuanzhu/Biomedical/data/ddi2013ms/test.tsv', sep='\t')


#######################################################定义模型参数#########################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
num_label=5


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
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
        label = label_map[label_str]

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

# 定义模型参数
max_length = 300
batch_size = 8

# 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
dev_data_loader = create_data_loader(df_dev, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)


# #输出data_loader
# for batch in train_data_loader:
#     print(batch)


#####################################################定义模型####################################################

# class YModel(nn.Module):
#     def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
#         super(YModel, self).__init__()
#         self.Bert_1 = BertModel.from_pretrained(model_name_1)
#         self.Bert_2 = BertModel.from_pretrained(model_name_2)
#         self.Bert_3 = BertModel.from_pretrained(model_name_3)

#         # Freeze BERT parameters
#         for param in self.Bert_1.parameters():
#             param.requires_grad = False
#         for param in self.Bert_2.parameters():
#             param.requires_grad = False
#         for param in self.Bert_3.parameters():
#             param.requires_grad = False

#         self.fc_1 = nn.Linear(768, 128)
#         self.fc_2 = nn.Linear(768, 128)
#         self.fc_3 = nn.Linear(768, 128)

#         # self.attention_1 = nn.MultiheadAttention(embed_dim=128, num_heads=2)
#         # self.attention_2 = nn.MultiheadAttention(embed_dim=128, num_heads=2)
#         # self.attention_3 = nn.MultiheadAttention(embed_dim=128, num_heads=2)
#         self.attention = nn.MultiheadAttention(embed_dim=384, num_heads=1) # 不会使用

#         self.fc = nn.Linear(128 * 3, num_labels)
#         self.loss_func = nn.CrossEntropyLoss()

#     def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
#         output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]
#         output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]
#         output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]
#         # Apply linear layer to each output
#         output_1 = self.fc_1(output_1[:, 0, :])  # Assuming you're taking the first token's output
#         output_2 = self.fc_2(output_2[:, 0, :])
#         output_3 = self.fc_3(output_3[:, 0, :])

#         # Apply multi-head attention
#         output_1, _ = self.attention_1(output_1.unsqueeze(0), output_1.unsqueeze(0), output_1.unsqueeze(0))
#         output_2, _ = self.attention_2(output_2.unsqueeze(0), output_2.unsqueeze(0), output_2.unsqueeze(0))
#         output_3, _ = self.attention_3(output_3.unsqueeze(0), output_3.unsqueeze(0), output_3.unsqueeze(0))

#         # Concatenate outputs
#         concatenated_output = torch.cat((output_1.squeeze(0), output_2.squeeze(0), output_3.squeeze(0)), dim=1)

#         # 这里应该有注意力

#         # Final classification layer
#         logits = self.fc(concatenated_output)

#         if labels is not None:
#             loss = self.loss_func(logits, labels)
#             return loss,logits
#         else:
#             return None,logits


class YModel(nn.Module):
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
        super(YModel, self).__init__()
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # Freeze BERT parameters
        # for param in self.Bert_1.parameters():
        #     param.requires_grad = False
        # for param in self.Bert_2.parameters():
        #     param.requires_grad = False
        # for param in self.Bert_3.parameters():
        #     param.requires_grad = False

        self.fc_1 = nn.Linear(768, 128)
        self.fc_2 = nn.Linear(768, 128)
        self.fc_3 = nn.Linear(768, 128)
        self.attention = nn.MultiheadAttention() 
        self.fc = nn.Linear(128 * 3, num_labels)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
        output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]  # output_1 shape: (batch_size, sequence_length, hidden_size)
        output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # output_2 shape: (batch_size, sequence_length, hidden_size)
        output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]  # output_3 shape: (batch_size, sequence_length, hidden_size)

        # Apply linear transformation
        output_1 = torch.tanh(self.fc_1(output_1[:, 0]))  # output_1 shape: (batch_size, 128)
        output_2 = torch.tanh(self.fc_2(output_2[:, 0]))  # output_2 shape: (batch_size, 128)
        output_3 = torch.tanh(self.fc_3(output_3[:, 0]))  # output_3 shape: (batch_size, 128)

        # Concatenate the outputs
        combined_output = torch.cat((output_1, output_2, output_3), dim=1)  # combined_output shape: (batch_size, 128 * 3)

        # Apply fully connected layer
        logits = self.fc(combined_output)  # logits shape: (batch_size, num_labels)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss,logits
        else:
            return None,logits
        


'''
class YModel(nn.Module):
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
        super(YModel, self).__init__()
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        Freeze BERT parameters
        for param in self.Bert_1.parameters():
            param.requires_grad = False
        for param in self.Bert_2.parameters():
            param.requires_grad = False
        for param in self.Bert_3.parameters():
            param.requires_grad = False

        self.fc_1 = nn.Linear(768, 128)
        self.fc_2 = nn.Linear(768, 128)
        self.fc_3 = nn.Linear(768, 128)
        self.attention = nn.MultiheadAttention() 
        self.fc = nn.Linear(128 * 3, num_labels)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
        output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]  # output_1 shape: (batch_size, sequence_length, hidden_size)
        output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # output_2 shape: (batch_size, sequence_length, hidden_size)
        output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]  # output_3 shape: (batch_size, sequence_length, hidden_size)

        # Apply linear transformation
        output_1 = torch.tanh(self.fc_1(output_1[:, 0]))  # output_1 shape: (batch_size, 128)
        output_2 = torch.tanh(self.fc_2(output_2[:, 0]))  # output_2 shape: (batch_size, 128)
        output_3 = torch.tanh(self.fc_3(output_3[:, 0]))  # output_3 shape: (batch_size, 128)

        # Concatenate the outputs
        combined_output = torch.cat((output_1, output_2, output_3), dim=1)  # combined_output shape: (batch_size, 128 * 3)

        # Apply fully connected layer
        logits = self.fc(combined_output)  # logits shape: (batch_size, num_labels)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss,logits
        else:
            return None,logits
'''

# best model
class YModel(nn.Module):
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
        super(YModel, self).__init__()
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # Freeze BERT parameters
        # for param in self.Bert_1.parameters():
        #     param.requires_grad = False
        # for param in self.Bert_2.parameters():
        #     param.requires_grad = False
        # for param in self.Bert_3.parameters():
        #     param.requires_grad = False

        self.fc_1 = nn.Linear(768, 128)
        self.fc_2 = nn.Linear(768, 128)
        self.fc_3 = nn.Linear(768, 128)
        self.attention = nn.MultiheadAttention() 
        self.fc = nn.Linear(128 * 3, num_labels)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
        output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]  # output_1 shape: (batch_size, sequence_length, hidden_size)
        output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # output_2 shape: (batch_size, sequence_length, hidden_size)
        output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]  # output_3 shape: (batch_size, sequence_length, hidden_size)

        # Apply linear transformation
        output_1 = torch.tanh(self.fc_1(output_1[:, 0]))  # output_1 shape: (batch_size, 128)
        output_2 = torch.tanh(self.fc_2(output_2[:, 0]))  # output_2 shape: (batch_size, 128)
        output_3 = torch.tanh(self.fc_3(output_3[:, 0]))  # output_3 shape: (batch_size, 128)

        # Concatenate the outputs
        combined_output = torch.cat((output_1, output_2, output_3), dim=1)  # combined_output shape: (batch_size, 128 * 3)

        # Apply fully connected layer
        logits = self.fc(combined_output)  # logits shape: (batch_size, num_labels)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss,logits
        else:
            return None,logits
        

# class YModel(nn.Module):
#     def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
#         super(YModel, self).__init__()
#         self.Bert_1 = BertModel.from_pretrained(model_name_1)
#         self.Bert_2 = BertModel.from_pretrained(model_name_2)
#         self.Bert_3 = BertModel.from_pretrained(model_name_3)

#         self.fc_1 = nn.Linear(768, 128)
#         self.fc_2 = nn.Linear(768, 128)
#         self.fc_3 = nn.Linear(768, 128)
#         self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=1)  # Self-attention layer
#         self.fc = nn.Linear(128, num_labels)
#         self.loss_func = nn.CrossEntropyLoss()

#     def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
#         output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]  # output_1 shape: (batch_size, sequence_length, hidden_size)
#         output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # output_2 shape: (batch_size, sequence_length, hidden_size)
#         output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]  # output_3 shape: (batch_size, sequence_length, hidden_size)

#         output_1 = torch.tanh(self.fc_1(output_1[:, 0]))  # output_1 shape: (batch_size, 128)
#         output_2 = torch.tanh(self.fc_2(output_2[:, 0]))  # output_2 shape: (batch_size, 128)
#         output_3 = torch.tanh(self.fc_3(output_3[:, 0]))  # output_3 shape: (batch_size, 128)

#         # Concatenate the outputs
#         combined_output = torch.stack([output_1, output_2, output_3], dim=1)  # combined_output shape: (batch_size, 3, 128)

#         # Apply self-attention
#         combined_output = combined_output.permute(1, 0, 2)  # Change the shape for self-attention layer
#         combined_output, _ = self.attention(combined_output, combined_output, combined_output)  # combined_output shape: (3, batch_size, 128)
#         combined_output = combined_output.mean(dim=0)  # Average the attention output over all tokens
#         combined_output = combined_output.squeeze(0)  # Remove the extra dimension added by unsqueezing

#         # Apply fully connected layer
#         logits = self.fc(combined_output)  # logits shape: (batch_size, num_labels)

#         if labels is not None:
#             loss = self.loss_func(logits, labels)
#             return loss, logits
#         else:
#             return None, logits
        

import torch
import torch.nn as nn
from transformers import BertModel



# class SciBERTForDDI(nn.Module):
#     def __init__(self, model_path, num_labels):
#         super(SciBERTForDDI, self).__init__()
#         self.bert = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return outputs.loss, outputs.logits


# 实例化模型
num_labels = len(label_map)
model = YModel(model_path_1,model_path_2,model_path_3, num_labels)
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=1)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 训练模型
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    losses = []
    targets = []
    predictions = []

    for batch in data_loader:
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)
        input_ids_3 = batch['input_ids_3'].to(device)
        attention_mask_3 = batch['attention_mask_3'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        loss, logits = model(input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2,
                             input_ids_3=input_ids_3,attention_mask_3=attention_mask_3, labels=labels)


        predictions.extend(torch.argmax(logits, dim=1).tolist())
        targets.extend(labels.tolist())

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if scheduler:
        scheduler.step()

    return losses, targets, predictions

# 验证模型
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            input_ids_3 = batch['input_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2,
                             input_ids_3=input_ids_3,attention_mask_3=attention_mask_3, labels=labels)
            
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            targets.extend(labels.tolist())

            losses.append(loss.item())

    return losses, targets, predictions

# 训练模型
epochs = 10  # 训练轮数
best_accuracy = 0
print("start train ")
for epoch in range(epochs):
    train_losses, train_targets, train_predictions = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
    train_accuracy = accuracy_score(train_targets, train_predictions)
    
    dev_losses, dev_targets, dev_predictions = eval_model(model, dev_data_loader, loss_fn, device)
    dev_accuracy = accuracy_score(dev_targets, dev_predictions)
    # 计算四大指标
    dev_accuracy = accuracy_score(dev_targets, dev_predictions)
    dev_precision = precision_score(dev_targets, dev_predictions,average='weighted')
    dev_recall = recall_score(dev_targets, dev_predictions,average='weighted')
    dev_f1 = f1_score(dev_targets, dev_predictions,average='weighted')

    # 添加对测试集的评估
    test_losses, test_targets, test_predictions = eval_model(model, test_data_loader, loss_fn, device)
    # 计算四大指标
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_precision = precision_score(test_targets, test_predictions,average='weighted')
    test_recall = recall_score(test_targets, test_predictions,average='weighted')
    test_f1 = f1_score(test_targets, test_predictions,average='weighted')
    
    test_precision_micro = precision_score(test_targets, test_predictions,average='micro')
    test_recall_micro = recall_score(test_targets, test_predictions,average='micro')
    test_f1_micro = f1_score(test_targets, test_predictions,average='micro')
    # 计算混淆矩阵
    cm = confusion_matrix(test_targets, test_predictions)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()

    #print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {sum(train_losses)/len(train_losses)}, Train Accuracy: {train_accuracy}, Dev Loss: {sum(dev_losses)/len(dev_losses)}, Dev Accuracy: {dev_accuracy}, Test Loss: {sum(test_losses)/len(test_losses)}, Test Accuracy: {test_accuracy}')
    
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {sum(train_losses)/len(train_losses)}')
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Dev Loss: {sum(dev_losses)/len(dev_losses)}')
    print(f'Dev Accuracy: {dev_accuracy}')
    print(f'Dev Precision: {dev_precision}')
    print(f'Dev Recall: {dev_recall}')
    print(f'Dev F1 Score: {dev_f1}')
    print("--")
    print(f'Test Loss: {sum(test_losses)/len(test_losses)}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1 Score: {test_f1}')
    print("--")
    print(f'Test Precision: {test_precision_micro}')
    print(f'Test Recall: {test_recall_micro}')
    print(f'Test F1 Score: {test_f1_micro}')



    if dev_accuracy > best_accuracy:
        best_accuracy = dev_accuracy
        torch.save(model.state_dict(), 'model.pth')