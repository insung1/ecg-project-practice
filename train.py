# train.py

import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from config import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, HISTORY_PATH, CHECKPOINT_DIR

def load_model_weights(model, epoch):
    if epoch == 0:
        print("📦 새 모델로부터 시작합니다.")
        return
    path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ 모델 가중치 로드 완료: {path}")
    else:
        print(f"❌ 모델 파일 {path} 없음. 새로 시작합니다.")

def train_model(model, train_dataset, val_dataset, device, start_epoch=0):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch + 1, start_epoch + NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"))
            print(f"💾 모델 저장됨: model_epoch_{epoch}.pth")

    return history

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds_all.extend((outputs >= 0.5).float().cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    return acc, avg_loss

def save_history(history, start_epoch):
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            full_history = json.load(f)
    else:
        full_history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for key in history:
        full_history[key].extend(history[key])

    with open(HISTORY_PATH, 'w') as f:
        json.dump(full_history, f, indent=2)

    return full_history
