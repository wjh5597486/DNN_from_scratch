# Neural Network from scratch

## 1. Propagation
![img_3.png](images/img_3.png)   
- With one batch
  - The first layer   
  ![img_5.png](images/img_5.png)
  - The second layer   
  ![img_6.png](images/img_6.png)  



- 2 size batch
  - The first layer  
  ![img_7.png](images/img_7.png)
  - The second layer  
  ![img_8.png](images/img_8.png)

    

## 2. Backpropagation

Equation for updating weights  
![img_9.png](images/img_9.png)  (Alpha is Learning rate.)  

The Equation for the first layer can be expressed the following.  
![img_10.png](images/img_10.png)    

### For updating the second layer, the matrix bellow need to be computed.
![img.png](images/img.png)  
where L is Loss, z^hat is the answer and the loss function is MSE. 

![img_1.png](images/img_1.png)

what we need is simply computed by calculating  **transpose(INPUT) * LOSS**


### For the first layer
![img_2.png](images/img_2.png)  
![img_4.png](images/img_4.png)

we will see the first row, first column element.
![img_11.png](images/img_11.png)

the other elements is computed respectively.
![img_12.png](images/img_12.png)

![img_13.png](images/img_13.png)

### The conclusion is that we can update weights with input, output and loss matrix.