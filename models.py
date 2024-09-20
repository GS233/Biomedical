import torch
import torch.nn as nn
from transformers import BertModel
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