def Confusion_Matrix(model,input,actual):
"""
model  : your pytorch model or each framework model
input  : model's input for example batch of images
actual : input's label
"""

    confusion_matrix  = np.zeros(9)
    with torch.no_grad():
        output = model(input)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1) # Network's output 
    actual = actual.numpy()    # Real      output

    for i in range(len(actual)):
        for j in range(9):
            n = j
            for k in range(3):
                if predictions[i] == k:
                    n += (3 * k + 1)
                    if k == 0:
                        if actual[i] == k:
                            confusion_matrix[n - 1] += 1    
                        elif actual[i] == k + 1:
                            confusion_matrix[n] += 1
                        elif actual[i] == k + 2:
                            confusion_matrix[n + 1] += 1    
                    elif k == 1:
                        if actual[i] == k - 1:
                            confusion_matrix[n - 1] += 1
                        elif actual[i] == k:
                            confusion_matrix[n] += 1
                        elif actual[i] == k + 1:
                            confusion_matrix[n + 1] += 1
                    elif k == 2:
                        if actual[i] == k:
                            confusion_matrix[n + 1] += 1
                        elif actual[i] == k - 1:
                            confusion_matrix[n] += 1
                        elif actual[i] == k - 2:
                            confusion_matrix[n - 1] += 1

            break        
    return confusion_matrix.reshape(3,3),predictions,actual 
