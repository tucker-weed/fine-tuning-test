import pandas as pd
import numpy.random as npr

SEED = 42
TRAIN_TEST_SPLIT = 0.95
BALANCE_DATASET = False
npr.seed(SEED)

data_path = "data/test.csv"

relevant_columns = ["IncidentDate", "IncidentDescription", "AverageWeeklyWage",
                    "ClaimantAge_at_DOI", "Gender", "ClaimantType",
                    "InjuryCause", "InjuryNature", "BodyPartRegion", "BodyPart"]
EMPTY_COLUMN = "not available"

def substitute_empty_columns(row):
    for column in relevant_columns:
        if row[column] == "#VALUE!" or str(row[column]) == "nan" or str(row[column]) == "NULL" or row[column] == "Not Available":
            row[column] = EMPTY_COLUMN
    return row

def build_input_string(row):
    input_string = f"Classify the following insurance claim text data into one of the following classes: [approved, denied].\n\nInsurance Claim Text Data:\n\n"

    for column in relevant_columns:
        input_string += f"{column}: {str(row[column]).lower()}\n"

    return input_string

def build_label_string(row):
    if str(row["IsDenied"]) == "0":
        return "approved"
    else:
        return "denied"

def preprocess(balance_dataset=True):
    claims_df = pd.read_csv(data_path)
    claims_df = claims_df.sample(frac=1, random_state=SEED)
    input_strings = []
    label_strings = []

    if balance_dataset:
        denied_claims = claims_df[claims_df["IsDenied"] == True]
        approved_claims = claims_df[claims_df["IsDenied"] == False]
        approved_claims = approved_claims.sample(frac=1, random_state=SEED)
        approved_claims = approved_claims.head(len(denied_claims))

        for denied_row, approved_row in zip(denied_claims.iterrows(), approved_claims.iterrows()):
            denied_row = denied_row[1]
            approved_row = approved_row[1]
            approved_row = substitute_empty_columns(approved_row)
            denied_row = substitute_empty_columns(denied_row)
            input_strings.append(build_input_string(approved_row))
            label_strings.append(build_label_string(approved_row))
            input_strings.append(build_input_string(denied_row))
            label_strings.append(build_label_string(denied_row))
    else:
        max_count = 10000
        total = 0
        total_approved_count = 0
        for row in claims_df.iterrows():
            total+= 1
            row = row[1]
            if row['IsDenied'] == 0:
                total_approved_count += 1
            row = substitute_empty_columns(row)
            input_strings.append(build_input_string(row))
            label_strings.append(build_label_string(row))
        
        total_dataset_percent_approved = total_approved_count / total
        print("Percent of data is claim approved =", total_dataset_percent_approved)

        input_strings = input_strings[:max_count]
        label_strings = label_strings[:max_count]

    train_size = int(TRAIN_TEST_SPLIT * len(input_strings))
    train_input = input_strings[:train_size]
    test_input = input_strings[train_size:]
    train_labels = label_strings[:train_size]
    test_labels = label_strings[train_size:]

    permutation = npr.permutation(len(train_input))
    train_input = [train_input[i] for i in permutation]
    train_labels = [train_labels[i] for i in permutation]

    permutation = npr.permutation(len(test_input))
    test_input = [test_input[i] for i in permutation]
    test_labels = [test_labels[i] for i in permutation]
    
    return train_input, train_labels, test_input, test_labels

def peek_data():
    X, Y, test_input, test_labels = preprocess(BALANCE_DATASET)
    print(X[1], Y[1])
    print("\nTrain examples:", len(X), "| Test examples:", len(test_input))

if __name__ == '__main__':
    peek_data()
