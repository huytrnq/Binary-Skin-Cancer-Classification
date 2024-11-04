import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Function to convert probabilities to Dempster-Shafer evidence (beliefs)
def get_evidence_from_probabilities(probs):
    evidence = {
        "A": probs[0],  # Belief for class A
        "B": probs[1],  # Belief for class B
        "uncertainty": 1 - sum(probs)  # Remaining uncertainty
    }
    return evidence

# Dempster-Shafer combination rule for combining evidence
def dempster_shafer_combination(evidence_list):
    combined_evidence = evidence_list[0]
    for evidence in evidence_list[1:]:
        new_combined_evidence = {}

        for hypo_i, belief_i in combined_evidence.items():
            for hypo_j, belief_j in evidence.items():
                if hypo_i == hypo_j:  # Combine beliefs for the same hypothesis
                    new_combined_evidence[hypo_i] = new_combined_evidence.get(hypo_i, 0) + belief_i * belief_j
                elif hypo_i != "uncertainty" and hypo_j != "uncertainty":  # Handle conflict
                    new_combined_evidence["conflict"] = new_combined_evidence.get("conflict", 0) + belief_i * belief_j

        # Normalize to resolve conflict
        conflict = new_combined_evidence.get("conflict", 0)
        if conflict < 1:
            for hypo in combined_evidence:
                if hypo != "conflict":
                    new_combined_evidence[hypo] = new_combined_evidence.get(hypo, 0) / (1 - conflict)
            new_combined_evidence.pop("conflict", None)

        combined_evidence = new_combined_evidence
    return combined_evidence

# Ensemble prediction function for models with different feature subsets
def dst_ensemble_predict(models, feature_sets):
    """
    Uses Dempster-Shafer Theory to ensemble predictions from multiple models.
    Args:
        models (list): List of trained models.
        feature_sets (list of np.ndarray): List of feature sets, each corresponding to a model.
    Returns:
        list: Final predictions after combining evidence.
    """
    final_predictions = []

    # Assume all feature sets have the same number of samples
    n_samples = feature_sets[0].shape[0]
    
    for i in range(n_samples):
        evidence_list = []

        # Get evidence from each model based on its specific feature subset
        for model, features in zip(models, feature_sets):
            probs = model.predict_proba(features[i, :].reshape(1, -1))[0]
            evidence = get_evidence_from_probabilities(probs)
            evidence_list.append(evidence)

        # Combine evidence using Dempster's Rule of Combination
        combined_evidence = dempster_shafer_combination(evidence_list)
        
        # Predict the class with the highest combined belief
        predicted_class = max(["A", "B"], key=lambda x: combined_evidence.get(x, 0))
        final_predictions.append(predicted_class)

    return final_predictions

# Evaluation function to assess DST ensemble performance
def evaluate_dst_ensemble(models, feature_sets, y_test):
    """
    Evaluates the DST ensemble on test data with models trained on different feature sets.
    Args:
        models (list): List of trained models.
        feature_sets (list of np.ndarray): List of test feature sets, each corresponding to a model.
        y_test (array-like): True labels for the test set.
    """
    # Get predictions from the DST ensemble
    ensemble_predictions = dst_ensemble_predict(models, feature_sets)

    # Map string labels to numeric for metric consistency, if necessary
    label_map = {"A": 0, "B": 1}
    numeric_predictions = [label_map[pred] for pred in ensemble_predictions]

    # Calculate metrics
    accuracy = accuracy_score(y_test, numeric_predictions)
    precision = precision_score(y_test, numeric_predictions, pos_label=0)
    recall = recall_score(y_test, numeric_predictions, pos_label=0)
    f1 = f1_score(y_test, numeric_predictions, pos_label=0)
    report = classification_report(y_test, numeric_predictions, target_names=["A", "B"])

    # Print metrics
    print("DST Ensemble Accuracy:", accuracy)
    print("DST Ensemble Precision:", precision)
    print("DST Ensemble Recall:", recall)
    print("DST Ensemble F1 Score:", f1)
    print("\nClassification Report:\n", report)

    return numeric_predictions
