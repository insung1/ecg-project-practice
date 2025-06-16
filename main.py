# main.py

import torch
import os
from config import NUM_EPOCHS, CHECKPOINT_DIR
from ecg_data import load_ecg_data, create_dataloaders
from model import ECG1DCNN
from train import train_model, save_history, load_model_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # 1. 데이터 로딩 (60/20/20)
    X_all, y_all = load_ecg_data()
    train_dataset, val_dataset, _ = create_dataloaders(X_all, y_all, batch_size=32)

    # 2. 이전 학습 에폭 불러오기
    start_epoch = get_current_epoch()
    print(f"🔁 현재까지 학습된 에폭: {start_epoch}회")

    # 3. 모델 정의 + 가중치 불러오기
    model = ECG1DCNN().to(device)
    load_model_weights(model, start_epoch)

    # 4. 학습 진행
    print(f"🚀 Epoch {start_epoch+1} ~ {start_epoch+NUM_EPOCHS} 학습 시작")
    history = train_model(model, train_dataset, val_dataset, device, start_epoch=start_epoch)

    # 5. 히스토리 저장
    save_history(history, start_epoch)

def get_current_epoch():
    import os
    import json
    from config import HISTORY_PATH
    if not os.path.exists(HISTORY_PATH):
        return 0
    with open(HISTORY_PATH, 'r') as f:
        data = json.load(f)
        return len(data['loss'])

if __name__ == "__main__":
    main()
