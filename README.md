# OperatorVision

![image](https://user-images.githubusercontent.com/125378481/219015040-049761d7-4acc-4878-9663-133652b61442.png)

The program should find the customer numbers and meter readings in the PNG image, and convert them to the STRING type.

![image](https://user-images.githubusercontent.com/125378481/219016471-7084a64e-de10-4251-bf84-167315c44124.png)
![image](https://user-images.githubusercontent.com/125378481/219016485-7ef3e06e-ab14-44bb-8c9b-396c2eeaa66f.png)

To implement this task, I trained four neural networks and labeled the training samples for them. The first neural network was trained to find customer numbers, the second - meter readings, then the found areas were cut into ones by the algorithm (15-76-100 -> 1,5,-,7,6,-,1,0,0) (017027 -> 0, 1, 7, 0, 2, 7), the third and fourth neural networks guessed the digits.
In the fourth (handwritten numbers), I managed to achieve 80% success in guessing the digit, and the percentage increased as the training sample increased.

Each of these four neural networks is a classical neural network with two hidden layers. I copied the structure of the neural network from Tariq Rashid's book "Creating a Neural Network." I rewrote from Python to C# and added a second hidden layer, built on the NuGet package MathNet.Numerics (matrices, linear algebra).

![image](https://user-images.githubusercontent.com/125378481/219018896-7e73204d-38ba-4eef-970d-abb365339336.png)

The basis of the project is my unfinished chess vision project. From it I took the Int_palette function, which converts an image with a standard pixel palette (1 million colors) into an int array with a very small color range (8,27,64),

![image](https://user-images.githubusercontent.com/125378481/219019334-a33b68d1-b9fc-4e17-b2be-79597376fc2e.png)

and the Perimeter function, which allows you to find individual elements in an image.

![image](https://user-images.githubusercontent.com/125378481/219019477-e6679487-33fc-4e03-96e6-6202650d2a93.png)
![image](https://user-images.githubusercontent.com/125378481/219019500-c8fbb15e-a4ad-4e62-859d-3259187c8789.png)
![image](https://user-images.githubusercontent.com/125378481/219019521-ad5de4cb-317a-4353-83db-3f37d74cab01.png)
![image](https://user-images.githubusercontent.com/125378481/219019530-9d06223c-56ff-4a44-8e78-23a36703da8a.png)

The program is divided into commented segments, which are uncommented to increase the training samples.

     //////The code is in the file OperatorVision/OperatorVision/Form1.cs

     //////async void button1_Click(object sender, EventArgs e)

This button opens the openFileDialog1 window with a Png image filter. The scanner's automatic flipping moved the stack of sheets left and right within 5 mm, which is why the borders of each page are different. 

     //////cut_image empty_lines
     
![image](https://user-images.githubusercontent.com/125378481/219022413-4b0d08a9-9de4-4d57-a6f5-1ec3027bb11c.png)
![image](https://user-images.githubusercontent.com/125378481/219022438-63729d7f-66f9-4eb7-b357-29469049de98.png)

(In the first image, the table starts at the 70th pixel, in the second at the 140th.)

![image](https://user-images.githubusercontent.com/125378481/219023258-73d4035e-fe95-4422-9629-ac8364dce0fa.png)
![image](https://user-images.githubusercontent.com/125378481/219023275-e5f89074-4161-4e26-829d-2e3a58e9823a.png)

(The same pages, but cut)

The Int_palette function converts an image into an int array, where the numbers 1 to 8 represent a specific range of colors.
(1 – white, 4 – red, 8 – black)
The cut_empty function cuts all empty (> 90% white) rows and columns from the pixel_array until the first time this condition fails, overwriting the array with the new number of rows and columns. The image is then saved back to PNG.

     //////cut_image

Now the horizontal coordinates of the customer numbers and meter readings are constant. The pages are cut into strips with areas where customer numbers and possible meter readings are located. Sometimes, when flipping, the stack of sheets rotated by a couple of degrees, due to which the beginning varies within 5-10 pixels, so cutting is done with a small margin.

![image](https://user-images.githubusercontent.com/125378481/219024477-6561d3ec-e773-4f4a-8889-407defcd3bb3.png)
![image](https://user-images.githubusercontent.com/125378481/219024517-7709d5b5-75dd-4244-bf5e-0bb00819a46f.png)

     //////cut_lines
The strips of customer numbers and meter readings are cut into lines using the cut_lines function. The height of the rows of the meter reading strips is slightly higher than for customer numbers, which allows you to remove most of the unnecessary rows.
 
![image](https://user-images.githubusercontent.com/125378481/219025917-09f0146c-f254-440b-8a2f-f01e4a775f55.png)
..........
![image](https://user-images.githubusercontent.com/125378481/219025931-df1988d7-1d09-4805-a892-7c0ea2eda153.png)

![image](https://user-images.githubusercontent.com/125378481/219025949-b86522fa-d0f7-417d-a7b0-f81991bb5b65.png)
..........
![image](https://user-images.githubusercontent.com/125378481/219025967-166e5440-dbb3-49e9-aadb-bbbb87386645.png)

![image](https://user-images.githubusercontent.com/125378481/219026091-f7751aa4-b3be-42a2-a558-dce7d9a5e3c4.png)
..........
![image](https://user-images.githubusercontent.com/125378481/219026121-b15b3aa9-9809-44a0-a284-32a6775c57b9.png)

![image](https://user-images.githubusercontent.com/125378481/219026281-3f85dbf5-4c9f-41c4-84e4-750d4ab8f8b8.png)
..........
![image](https://user-images.githubusercontent.com/125378481/219026319-900528ef-9776-4779-9d8d-b131c44cae1a.png)

![image](https://user-images.githubusercontent.com/125378481/219026341-74a84d89-7959-4c0d-9816-514d3b451f5f.png)
..........
![image](https://user-images.githubusercontent.com/125378481/219026365-6e6cd2f8-3612-421a-8d45-55a6e9d426f9.png)

     //////Marking
After the lines have been collected, they need to be marked. In these gifs you can see how the training sample was formed for an earlier version of the program.

![1](https://user-images.githubusercontent.com/125378481/219029829-0a876fd9-702b-4399-81d7-4cd14826d31a.gif)

![3](https://user-images.githubusercontent.com/125378481/219032877-d5d51f32-c4f0-42d6-83da-986db7094b85.gif)

![image](https://user-images.githubusercontent.com/125378481/219030636-847756fc-4354-4f70-9855-59f7c5dddfa8.png)
 
(not meter readings are marked "M")

     //////delete quadrilaterals
The table boundaries do not allow the algorithm to divide the meter readings by ones, the delete_quadrilaterals function deletes boundaries.

![image](https://user-images.githubusercontent.com/125378481/219032427-29b0fa74-d032-4bcc-abc8-e100c804cf66.png)

![image](https://user-images.githubusercontent.com/125378481/219032458-e9b6c94d-41cd-4f93-ad4d-1573f5ab1cdd.png)

![image](https://user-images.githubusercontent.com/125378481/219032481-f76dd5ef-f3cd-4695-a9a1-3a484e9f0349.png)

![image](https://user-images.githubusercontent.com/125378481/219032500-440b2e84-706c-4141-a69e-a3af352dab53.png)

![image](https://user-images.githubusercontent.com/125378481/219032521-7b7cb99b-4d6f-46d4-b57d-3ff74111edbf.png)

![image](https://user-images.githubusercontent.com/125378481/219032560-6e36c5d0-7624-4ab5-a2a2-86e0a56be6fe.png)

The numbers lose a lot of quality because of this function, which is the reason for the drop in the efficiency of the fourth neural network.

     //////txt
Marked images are overwritten in a txt file. Images are compressed to save resources and reduce training time. The cut_jpg function shortens columns and rows, overwriting the array.

![image](https://user-images.githubusercontent.com/125378481/219036400-06ea90d6-1f58-4af3-b035-0a739ec1cb61.png)

![image](https://user-images.githubusercontent.com/125378481/219036480-dc24fc6b-8000-4a94-89c0-59da28f3b7a5.png)

     //////first part
Other neural networks were trained in a similar way

The read_input function writes the training sample to an int array, the variable num is the number of all images written to this array. The training neural network will be tested on the last 200 images.
  
//for (int d1 = 0; d1 < 15; d1++)

This cycle looks for delta learning coefficient, changing other parameters of the neural network did not bring any results.
A one-dimensional double array converted from a two-dimensional(100х10) array is inserted into the Intelect2[0] neural network. In two hidden layers, a "thinking" process occurs, which forms a double array with one element, where the value of this element, tending to 0.01, says that there is no customer numbers in the inserted row, and the value, tending to 0.99, says that there is.

//for (int c = 0; c < 10000; c++)

The training sample is small, the number of its elements is not enough, but these elements can be used several times.
The Random() function trains better than a cycle that sequentially repeats all elements of the training sample.

// for (int c = num; c <num + 200; c++)

The efficiency of the neural network is calculated on the last 200 labeled images.
Then the results and weights of the network neurons are written to txt files.

![image](https://user-images.githubusercontent.com/125378481/219041985-4c884467-cc30-4d1e-b9db-66c2578d61f7.png)

(file with saved trained neural network)

     //////mechanic digits
I planned to identify printed digits by checking for correspondence with previously prepared tracings of digits, but this method failed. The scanned printed digit is often unique (scanned at low resolution), so I had to use neural networks.
The Perimeter function slices customer numbers into ones than the neural network learns to guess printed digits.

![image](https://user-images.githubusercontent.com/125378481/219044207-89665242-5542-4893-ae74-95d56daa6454.png)
![image](https://user-images.githubusercontent.com/125378481/219044226-3d912b13-fcd2-420e-b1de-e80db5bb9910.png)
![image](https://user-images.githubusercontent.com/125378481/219044246-48547c7c-b091-4903-a116-53383d4bf49e.png)

![image](https://user-images.githubusercontent.com/125378481/219044268-d342c1c2-b473-4b74-8476-89179dfa760e.png)
![image](https://user-images.githubusercontent.com/125378481/219044292-7148de3b-9662-4b77-a612-2403ef586abc.png)
![image](https://user-images.githubusercontent.com/125378481/219044315-1163d6e7-bcb4-4dd1-9f89-9eb7aac513bd.png)

![image](https://user-images.githubusercontent.com/125378481/219044344-346d42ae-da48-4e03-8b84-61a675fc936a.png)
![image](https://user-images.githubusercontent.com/125378481/219044356-2201f83c-9632-4136-ab8f-c136368adb62.png)
![image](https://user-images.githubusercontent.com/125378481/219044376-d7fe30d5-26ae-4da5-9ade-8c9bc1f2fcc9.png)

     //////second part
This neural network learns to determine meter readings. The neural network has a slightly lower efficiency (96-98%) than the one that works with customer numbers (99%+), but this does not hinder the combined program. Customer numbers and meter readings are parallel, false positives are filtered out because they do not have a customer number pair.

![image](https://user-images.githubusercontent.com/125378481/219045492-1872983d-ec93-4281-8912-58b1c974ed9d.png)
(the neural network incorrectly determined that this line is a meter reading)

     //////third part
The neural network learns to guess handwritten digits that are cut from the meter reading lines by the Perimeter function.
Initially, I used the MNIST dataset (60k labeled digits), but on the digits written by the controller, the neural network showed terrible results (<50%, although on the MNIST test dataset the percentage of successful guesses was about 98).I had to do the digit marking myself. With two thousand digit markings, I achieved 80 percent of successful guesses, and perhaps with an increase in the training sample, the percentage would reach the level of 90-95. However, I am sure that this is a dead end branch of development. It would be more appropriate to replace the area where the meter readings are written with a postal code.

![image](https://user-images.githubusercontent.com/125378481/219046882-bfcab13f-1cdc-424c-a768-2e7350994698.png)

     //////united
I don't have many scanned pages, training samples were formed from all of them, except for the last five. The combined program is tested on them.(video link
https://drive.google.com/file/d/1CkTENVJSMnnw9AY8-0b71aGU57q4PX2-/view?usp=sharing)
 
 

