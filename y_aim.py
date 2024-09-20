import torch
import torch.cuda
import pandas as pd
import torch.nn as nn
from torchvision.transforms import transforms
from transformers import BertModel, AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import *
from tools import *
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
model_path_1 = './model_path/medbert'
scibert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
scibert_model = AutoModel.from_pretrained(model_path_1)
model_path_2 = './model_path/biobert'
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path_2)
biobert_model = AutoModel.from_pretrained(model_path_2)
model_path_3 = './model_path/bluebert'
bluebert_tokenizer = AutoTokenizer.from_pretrained(model_path_3)
bluebert_model = AutoModel.from_pretrained(model_path_3)
print("model load")
df_train = pd.read_csv('./data/aimed/1/train.tsv', sep='\t')
# df_dev = pd.read_csv('./data/ChemProtMS/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/aimed/1/test.tsv', sep='\t')

#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
num_label=2



# 定义模型参数
max_length = 300
batch_size = 8

# 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
# dev_data_loader = create_data_loader(df_dev, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, scibert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)



# 实例化模型
num_labels = 2
model = YModel(model_path_1,model_path_2,model_path_3, num_labels)
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

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
epochs = 20  # 训练轮数
best_accuracy = 0
print("start train ")
for epoch in range(epochs):
    train_losses, train_targets, train_predictions = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
    train_accuracy = accuracy_score(train_targets, train_predictions)

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
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {sum(train_losses)/len(train_losses)}')
    print(f'Train Accuracy: {train_accuracy}')
    # print(f'Dev Loss: {sum(dev_losses)/len(dev_losses)}')
    # print(f'Dev Accuracy: {dev_accuracy}')
    # print(f'Dev Precision: {dev_precision}')
    # print(f'Dev Recall: {dev_recall}')
    # print(f'Dev F1 Score: {dev_f1}')
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
    # if dev_accuracy > best_accuracy:
    #     best_accuracy = dev_accuracy
    #     torch.save(model.state_dict(), 'model.pth')