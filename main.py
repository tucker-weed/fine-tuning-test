import vertexai
import time
from skllm.models.palm import PaLM
from vertexai.preview.language_models import TextGenerationModel
from preprocess import preprocess

vertexai.init(project="573610933227", location="us-central1")

BALANCE_DATASET = True

NEW_MODEL = False
TEST_MODEL = True
DO_BASELINE_TEST = False
BASE_MODEL_NAME = "text-bison@001"
tuned_model_name = "projects/573610933227/locations/us-central1/models/2833967031336828928"

X, Y, test_input, test_labels = preprocess(BALANCE_DATASET)

def majority_weight(input_data):
    labels = {}
    n_samples = 1
    for _ in range(n_samples):
        predicted_label = str(model.predict(input_data, temperature=0.0))
        labels[predicted_label] = labels.get(predicted_label, 0) + 1
    return max(labels, key=labels.get)

if NEW_MODEL:
    model = PaLM(n_update_steps=200)
    model.fit(X, Y)
    tuned_model_name = model.tuned_model_

if TEST_MODEL:
    base_model = TextGenerationModel.from_pretrained(BASE_MODEL_NAME)
    fine_tuned_model = TextGenerationModel.get_tuned_model(tuned_model_name)
    if DO_BASELINE_TEST:
        print("LOADING BASE MODEL:", BASE_MODEL_NAME)
        model = base_model
    else:
        print("LOADING TUNED MODEL:", tuned_model_name)
        model = fine_tuned_model

    classes = ["approved", "denied"]
    test_labels_without_blocked_labels = []
    predict_labels = []
    i = 0
    total_predicted_approved = 0
    total_denied_labels = 0
    total_correctly_predicted_denied = 0
    for input_data in test_input:
        can_continue = False
        while not can_continue:
            try:
                time.sleep(0.5)
                label = majority_weight(input_data)

                if label != "":
                    if label == classes[0]:
                        total_predicted_approved += 1
                    if test_labels[i] == classes[1]:
                        total_denied_labels += 1
                        if label == test_labels[i]:
                            total_correctly_predicted_denied += 1
                    test_labels_without_blocked_labels.append(test_labels[i])
                    predict_labels.append(str(label))
                else:
                    print("BLOCKED")
                i += 1
                print("Testing:", i, "/", len(test_input))
                can_continue = True
            except Exception:
                time.sleep(7)

    count_correct = 0.0
    for i in range(len(test_labels_without_blocked_labels)):
        if test_labels_without_blocked_labels[i] == predict_labels[i]:
            count_correct += 1.0
    
    total_dataset_percent_approved = total_predicted_approved / len(test_labels_without_blocked_labels)
    print("\nPercent of data is predicted claim approved =", total_dataset_percent_approved)
    print("Denied claim accuracy =", total_correctly_predicted_denied / total_denied_labels)

    print("\nACCURACY:", count_correct / float(len(test_labels_without_blocked_labels)))
