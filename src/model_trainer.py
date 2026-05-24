import torch
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    def __init__(self, model, device, criterion, optimizer, class_names=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.class_names = class_names

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.model.to(self.device)

    def train_one_epoch(self, train_loader):
        self.model.train()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            preds = outputs.argmax(dim=1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = running_corrects / total

        return val_loss, val_acc

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train - Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            print("-" * 25)

        return self.history

    def evaluate(self, test_loader):
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)

        if self.class_names is None:
            target_names = None
        else:
            target_names = [self.class_names[i] for i in range(len(self.class_names))]

        report = classification_report(
            all_labels,
            all_preds,
            target_names=target_names,
        )

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
