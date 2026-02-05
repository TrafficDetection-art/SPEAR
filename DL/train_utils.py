import torch
import gc
from transformers import Trainer
import torch.nn as nn
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def custom_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted',zero_division=0)
    return accuracy, precision, recall, f1

def transformer_metrics(p, accuracy_metric, precision_metric, recall_metric, f1_metric):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

def train_transformer_model(model_name, model_path, train_dataset, eval_dataset, training_args, accuracy_metric, precision_metric, recall_metric, f1_metric):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: transformer_metrics(p, accuracy_metric, precision_metric, recall_metric, f1_metric)
    )
    
    trainer.train()
    trainer.save_model(f"{model_name}")
    tokenizer.save_pretrained(f"{model_name}")
    del model
    torch.cuda.empty_cache()
    gc.collect()

def train_custom_model(model,model_name,train_dataloader, eval_dataloader, training_args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    model.to(device)
    for epoch in range(training_args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                eval_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy, precision, recall, f1 = custom_metrics(np.array(all_preds), np.array(all_labels))
        print(f"Epoch {epoch + 1}, Eval Loss: {eval_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

        # Eval logic...
    
    # 检查 models 目录是否存在，不存在则创建
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"创建目录: {models_dir}")

    # 保存模型到 ./models/ 目录，并使用 _model.pt 后缀
    save_path = os.path.join(models_dir, f"{model_name}_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
