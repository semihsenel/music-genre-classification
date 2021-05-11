from glob import glob
import pandas as pd
import numpy as np

outputs = glob("../Outputs/*.txt")

if __name__ == '__main__':
    for i in outputs:
        with open(i, "r") as f:
            df = []
            lines = f.readlines()
            flag = True
            for line in lines:
                if "------>" in line:
                    if flag:
                        model = line[:line.find("------>")-1]
                    else:
                        model += line.strip()
                        model = model[:model.find("------>")-1]
                        flag = True
                elif ("------>" not in line) and "," in line:
                    model = line
                    flag = False
                else:
                    if "accuracy" in line:
                        accuracy = line[line.find(":")+2:]
                        accuracy = str(round(float(accuracy),3))
                    elif "recall" in line:
                        recall = line[line.find(":")+2:]
                        recall = str(round(float(recall),3))
                    elif "precision" in line:
                        precision = line[line.find(":")+2:]
                        precision = str(round(float(precision),3))
                    elif "f1" in line:
                        f1 = line[line.find(":")+2:]
                        f1 = str(round(float(f1),3))
                        df.append([model, accuracy, recall, precision, f1])
            
            
            df = pd.DataFrame(df, columns=["Model", "Accuracy", "Recall", "Precision", "F1 Score"])
            filename = i.split("\\")[-1].replace(".txt",".xlsx")
            filename = "../Tables/" + filename
            writer = pd.ExcelWriter(filename)
            df.to_excel(writer, "scores", engine='xlsxwriter')
            writer.save()
            print("{} created succesfully".format(filename.split("/")[-1]))
                    
            
                
