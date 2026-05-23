import pandas as pd
import torch

from src.metrics import Metrics


class XAIExperiment:
    def __init__(
        self,
        model,
        device,
        class_names,
        clean_examples,
        explainer,
        perturbations,
        top_k_percent=0.10,
        deletion_steps=20,
        baseline_value=-1.0,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.clean_examples = clean_examples
        self.selected_classes = sorted(clean_examples.keys())

        self.explainer = explainer
        self.perturbations = perturbations

        self.top_k_percent = top_k_percent
        self.deletion_steps = deletion_steps
        self.baseline_value = baseline_value

    def get_clean_tensor(self, class_id):
        return self.clean_examples[class_id].unsqueeze(0).to(self.device)

    def get_perturbed_tensor(self, clean_tensor, class_id, perturbation_type, value):
        if perturbation_type == "gaussian":
            return self.perturbations.gaussian(
                input_tensor=clean_tensor,
                sigma=value,
            )

        if perturbation_type == "fgsm":
            return self.perturbations.fgsm(
                input_tensor=clean_tensor,
                target_class=class_id,
                epsilon=value,
            )

        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    def get_prediction_info(self, input_tensor, target_class):
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)

            pred = output.argmax(dim=1).item()
            pred_conf = probs[0, pred].item()
            target_conf = probs[0, target_class].item()

        return {
            "pred": pred,
            "pred_conf": pred_conf,
            "target_conf": target_conf,
        }

    def run_similarity_metrics(self, perturbation_type, values):
        results = []

        for value in values:
            print(f"Processing {perturbation_type}={value}")

            for class_id in self.selected_classes:
                clean_tensor = self.get_clean_tensor(class_id)

                perturbed_tensor = self.get_perturbed_tensor(
                    clean_tensor=clean_tensor,
                    class_id=class_id,
                    perturbation_type=perturbation_type,
                    value=value,
                )

                clean_attr = self.explainer.explain(
                    input_tensor=clean_tensor,
                    target_class=class_id,
                )

                perturbed_attr = self.explainer.explain(
                    input_tensor=perturbed_tensor,
                    target_class=class_id,
                )

                cosine = Metrics.cosine_similarity(
                    clean_attr,
                    perturbed_attr,
                )

                iou = Metrics.topk_iou(
                    clean_attr,
                    perturbed_attr,
                    top_k_percent=self.top_k_percent,
                )

                results.append(
                    {
                        "Perturbation": perturbation_type,
                        "Value": value,
                        "ClassID": class_id,
                        "ClassName": self.class_names[class_id],
                        "CosineSimilarity": cosine,
                        "TopK_IoU": iou,
                    }
                )

        return pd.DataFrame(results)

    def run_deletion_auc(self, perturbation_type, values):
        results = []

        for value in values:
            print(f"Processing {perturbation_type}={value}")

            for class_id in self.selected_classes:
                clean_tensor = self.get_clean_tensor(class_id)

                perturbed_tensor = self.get_perturbed_tensor(
                    clean_tensor=clean_tensor,
                    class_id=class_id,
                    perturbation_type=perturbation_type,
                    value=value,
                )

                clean_info = self.get_prediction_info(
                    input_tensor=clean_tensor,
                    target_class=class_id,
                )

                perturbed_info = self.get_prediction_info(
                    input_tensor=perturbed_tensor,
                    target_class=class_id,
                )

                clean_attr = self.explainer.explain(
                    input_tensor=clean_tensor,
                    target_class=class_id,
                )

                perturbed_attr = self.explainer.explain(
                    input_tensor=perturbed_tensor,
                    target_class=class_id,
                )

                clean_auc = Metrics.deletion_auc(
                    model=self.model,
                    input_tensor=clean_tensor,
                    attribution_map=clean_attr,
                    target_class=class_id,
                    steps=self.deletion_steps,
                    baseline_value=self.baseline_value,
                )

                perturbed_auc = Metrics.deletion_auc(
                    model=self.model,
                    input_tensor=perturbed_tensor,
                    attribution_map=perturbed_attr,
                    target_class=class_id,
                    steps=self.deletion_steps,
                    baseline_value=self.baseline_value,
                )

                results.append(
                    {
                        "Perturbation": perturbation_type,
                        "Value": value,
                        "ClassID": class_id,
                        "ClassName": self.class_names[class_id],
                        "CleanPred": clean_info["pred"],
                        "PerturbedPred": perturbed_info["pred"],
                        "PredictionChanged": clean_info["pred"]
                        != perturbed_info["pred"],
                        "CleanPredConfidence": clean_info["pred_conf"],
                        "PerturbedPredConfidence": perturbed_info["pred_conf"],
                        "CleanTargetConfidence": clean_info["target_conf"],
                        "PerturbedTargetConfidence": perturbed_info["target_conf"],
                        "DeletionAUC_Clean": clean_auc,
                        "DeletionAUC_Perturbed": perturbed_auc,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def make_pivot(df, value_column):
        return df.pivot(
            index="Value",
            columns="ClassName",
            values=value_column,
        )

    @staticmethod
    def summarize_similarity(df):
        return (
            df.groupby("Value")
            .agg(
                CosineMean=("CosineSimilarity", "mean"),
                CosineStd=("CosineSimilarity", "std"),
                TopKIoUMean=("TopK_IoU", "mean"),
                TopKIoUStd=("TopK_IoU", "std"),
            )
            .reset_index()
        )

    @staticmethod
    def summarize_deletion(df):
        return (
            df.groupby("Value")
            .agg(
                CleanAUCMean=("DeletionAUC_Clean", "mean"),
                CleanAUCStd=("DeletionAUC_Clean", "std"),
                PerturbedAUCMean=("DeletionAUC_Perturbed", "mean"),
                PerturbedAUCStd=("DeletionAUC_Perturbed", "std"),
                PerturbedTargetConfidenceMean=("PerturbedTargetConfidence", "mean"),
                PredictionChanges=("PredictionChanged", "sum"),
            )
            .reset_index()
        )
