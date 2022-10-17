#!/usr/bin/env python3
# Imports
import os
import sys
import yaml

# The functions
def check_load_data():
    if os.path.exists("/data/train.csv") and os.path.exists("/data/test.csv"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Initial dataset not saved please check function code"}))

def check_prep():
    if os.path.exists("/data/train_norm.csv") and os.path.exists("/data/test_norm.csv") and os.path.exists("/data/train_pro.csv") and os.path.exists("/data/test_pro.csv"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Preprocessed dataset not saved please check function code"}))

def check_split():
    if os.path.exists("/data/X_train.csv") and os.path.exists("/data/X_test.csv") and os.path.exists("/data/y_train.csv") and os.path.exists("/data/y_test.csv"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Splited train test set not saved please check function code"}))

def check_train():
    if os.path.exists("/data/model.pkl"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Model not saved please check function code"}))
    
def check_predict():
    if os.path.exists("/data/result.csv"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Result not saved please check function code"}))

def check_mutual():
    if os.path.exists("/data/feature_mutual_info.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_corr():
    if os.path.exists("/data/feature_corrlation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_Pclass():
    if os.path.exists("/data/Pclass_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_Bridge():
    if os.path.exists("/data/Bridge_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_Title():
    if os.path.exists("/data/Title_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_age():
    if os.path.exists("/data/age_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_sex():
    if os.path.exists("/data/Sex_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_fare():
    if os.path.exists("/data/bridge_fare.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

def check_Parch():
    if os.path.exists("/data/Parch_relation.png"):
        print(yaml.dump({"output":"Function runs successfully"}))
    else:
        print(yaml.dump({"output":"Figure not saved please check function code"}))

if __name__ == '__main__':
    if len(sys.argv) != 2 or (sys.argv[1] != "check_load_data" and sys.argv[1] != "check_prep" and sys.argv[1] != "check_split" and sys.argv[1] != "check_train" and sys.argv[1] != "check_predict" and sys.argv[1] != "check_mutual" and sys.argv[1] != "check_corr" and sys.argv[1] != "check_Pclass" and sys.argv[1] != "check_Bridge" and sys.argv[1] != "check_Title" and sys.argv[1] != "check_age" and sys.argv[1] != "check_sex" and sys.argv[1] != "check_fare" and sys.argv[1] != "check_Parch"):
        print(f"Usage: {sys.argv[0]} check_load_data|check_prep|check_split|check_train|check_predict|check_mutual|check_Bridge|check_Title|check_age|check_sex|check_fare|check_Parch")
        exit(1)

    command = sys.argv[1]
    if command == "check_load_data":
        check_load_data()
    elif command == "check_prep":
        check_prep()
    elif command == "check_split":
        check_split()
    elif command == "check_train":
        check_train()
    elif command == "check_predict":
        check_predict()
    elif command == "check_mutual":
        check_mutual()
    elif command == "check_corr":
        check_corr()
    elif command == "check_Pclass":
        check_Pclass()
    elif command == "check_Bridge":
        check_Bridge()
    elif command == "check_Title":
        check_Title()
    elif command == "check_age":
        check_age()
    elif command == "check_sex":
        check_sex()
    elif command == "check_fare":
        check_fare()
    elif command == "check_Parch":
        check_Parch()

#Done
