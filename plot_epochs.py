# plot_epochs.py

import json
import matplotlib.pyplot as plt

HISTORY_PATH = "history.json"

def load_history():
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)

def plot_history(history, start_epoch=0, end_epoch=None, mode=2):
    if end_epoch is None:
        end_epoch = len(history["loss"])

    epochs = list(range(start_epoch + 1, end_epoch + 1))

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    if mode in (0, 2):
        plt.plot(epochs, history["loss"][start_epoch:end_epoch], label="Train Loss")
    if mode in (1, 2):
        plt.plot(epochs, history["val_loss"][start_epoch:end_epoch], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss ({start_epoch+1}~{end_epoch})")
    plt.legend()
    plt.grid(True)

    # x축 눈금을 10단위로 설정
    tick_interval = 10
    xticks = list(range(start_epoch + 1, end_epoch + 1, tick_interval))
    plt.xticks(xticks)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    if mode in (0, 2):
        plt.plot(epochs, history["acc"][start_epoch:end_epoch], label="Train Acc")
    if mode in (1, 2):
        plt.plot(epochs, history["val_acc"][start_epoch:end_epoch], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy ({start_epoch+1}~{end_epoch})")
    plt.legend()
    plt.grid(True)

    # x축 눈금을 10단위로 설정
    plt.xticks(xticks)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history = load_history()

    # ✅ 범위 설정
    start_epoch = 0
    end_epoch = 130

    # ✅ 모드 설정
    # 0 = train만, 1 = val만, 2 = 둘 다
    mode = 0

    plot_history(history, start_epoch, end_epoch, mode)
