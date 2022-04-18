with open("asama1.txt","r") as file:
    lines = file.readlines()
    correct = 0
    TP = 0
    TN = 0
    FP = 0 
    FN = 0

    for i,line in enumerate(lines[1:]):
        label = line[len(line)-2:len(line)-1]
       

        if line[:-1].split(" ")[0][:2] == "IN":
            if label == "0":
                correct +=1
                TN += 1
            else:
                FP += 1
                

        else:
            if label == "1":
                correct +=1
                TP += 1
            else:
                FN += 1

    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    
    
print("Correct Classification: ", correct)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
print("Average: {0:.3f}".format((sensitivity+specificity)/2))
                   
