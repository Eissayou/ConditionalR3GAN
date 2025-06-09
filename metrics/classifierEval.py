import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classifier(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred)
    return report, cm