import pandas as pd
import torch

from src.config import ExperimentConfig
from src.metrics import Metrics
from src.perturbations import Perturbations


class XAIExperiment:
    def __init__(self, model, device, class_names, clean_examples, explainer):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.clean_examples = clean_examples
        self.selected_classes = sorted(clean_examples.keys())

        self.explainer = explainer
        self.perturbations = Perturbations(self.model, self.device)
        self.metrics = Metrics()

        self.experiment_config = ExperimentConfig()

    def get_clean_tensor(self, class_id, to_device=True):
        tensor = self.clean_examples[class_id].unsqueeze(0)

        if to_device:
            tensor = tensor.to(self.device)

        return tensor

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

    def iter_clean_examples(self):
        for class_id in self.selected_classes:
            for example_idx, image in enumerate(self.clean_examples[class_id]):
                yield class_id, example_idx, image.unsqueeze(0).to(self.device)

    def run_similarity_metrics(self, perturbation_type, values):
        results = []

        clean_attr_cache = {}

        for class_id, example_idx, clean_tensor in self.iter_clean_examples():
            clean_attr_cache[(class_id, example_idx)] = self.explainer.explain(
                input_tensor=clean_tensor,
                target_class=class_id,
            )["abs"]

        for value in values:
            print(f"Processing {perturbation_type}={value}")

            for class_id, example_idx, clean_tensor in self.iter_clean_examples():
                perturbed_tensor = self.get_perturbed_tensor(
                    clean_tensor=clean_tensor,
                    class_id=class_id,
                    perturbation_type=perturbation_type,
                    value=value,
                )

                clean_attr = clean_attr_cache[(class_id, example_idx)]

                perturbed_attr = self.explainer.explain(
                    input_tensor=perturbed_tensor,
                    target_class=class_id,
                )["abs"]

                cosine = self.metrics.cosine_similarity(
                    clean_attr,
                    perturbed_attr,
                )

                iou = self.metrics.topk_iou(
                    clean_attr,
                    perturbed_attr,
                    top_k_percent=self.experiment_config.top_k_percent,
                )

                results.append(
                    {
                        "Perturbation": perturbation_type,
                        "Value": value,
                        "ClassID": class_id,
                        "ClassName": self.class_names[class_id],
                        "ExampleID": example_idx,
                        "CosineSimilarity": cosine,
                        "TopK_IoU": iou,
                    }
                )

        return pd.DataFrame(results)

    def run_deletion_auc(self, perturbation_type, values):
        results = []

        clean_cache = {}

        for class_id, example_idx, clean_tensor in self.iter_clean_examples():
            with torch.no_grad():
                clean_output = self.model(clean_tensor)
                clean_pred = clean_output.argmax(dim=1).item()
                clean_target_conf = torch.softmax(clean_output, dim=1)[
                    0, class_id
                ].item()

            clean_attr = self.explainer.explain(
                input_tensor=clean_tensor,
                target_class=class_id,
            )["deletion"]

            clean_auc = self.metrics.deletion_auc(
                model=self.model,
                input_tensor=clean_tensor,
                attribution_map=clean_attr,
                target_class=class_id,
                steps=self.experiment_config.deletion_steps,
                baseline_value=self.experiment_config.baseline_value,
            )

            clean_cache[(class_id, example_idx)] = {
                "pred": clean_pred,
                "target_conf": clean_target_conf,
                "auc": clean_auc,
            }

        for value in values:
            print(f"Processing {perturbation_type}={value}")

            for class_id, example_idx, clean_tensor in self.iter_clean_examples():
                perturbed_tensor = self.get_perturbed_tensor(
                    clean_tensor=clean_tensor,
                    class_id=class_id,
                    perturbation_type=perturbation_type,
                    value=value,
                )

                with torch.no_grad():
                    perturbed_output = self.model(perturbed_tensor)
                    perturbed_pred = perturbed_output.argmax(dim=1).item()
                    perturbed_target_conf = torch.softmax(
                        perturbed_output,
                        dim=1,
                    )[0, class_id].item()

                perturbed_attr = self.explainer.explain(
                    input_tensor=perturbed_tensor,
                    target_class=class_id,
                )["deletion"]

                perturbed_auc = self.metrics.deletion_auc(
                    model=self.model,
                    input_tensor=perturbed_tensor,
                    attribution_map=perturbed_attr,
                    target_class=class_id,
                    steps=self.experiment_config.deletion_steps,
                    baseline_value=self.experiment_config.baseline_value,
                )

                clean_data = clean_cache[(class_id, example_idx)]

                results.append(
                    {
                        "Perturbation": perturbation_type,
                        "Value": value,
                        "ClassID": class_id,
                        "ClassName": self.class_names[class_id],
                        "ExampleID": example_idx,
                        "PredictionChanged": clean_data["pred"] != perturbed_pred,
                        "CleanTargetConfidence": clean_data["target_conf"],
                        "PerturbedTargetConfidence": perturbed_target_conf,
                        "DeletionAUC_Clean": clean_data["auc"],
                        "DeletionAUC_Perturbed": perturbed_auc,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def make_pivot(df, value_column):
        return df.pivot_table(
            index="Value",
            columns="ClassName",
            values=value_column,
            aggfunc="mean",
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
                CleanTargetConfidenceMean=("CleanTargetConfidence", "mean"),
                PerturbedTargetConfidenceMean=("PerturbedTargetConfidence", "mean"),
                PredictionChanges=("PredictionChanged", "sum"),
            )
            .reset_index()
        )
