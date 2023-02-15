# OperatorVision
Програма, що автоматизує роботу оператора ЕОМ ділянки лінійних контролерів

Я комплексував з приводу величини свого портфоліо і не мав ідей для якихось проектів, тому вирішив влаштуватися на низькокваліфіковану роботу, щоб її автоматизувати. Більшість часу на цьому робочому місці я витрачав на перенесення з паперу в базу даних показань лічильника, написаних рукою контролера, який ходить із цими паперами по району.
![image](https://user-images.githubusercontent.com/125378481/219015040-049761d7-4acc-4878-9663-133652b61442.png)

Програма, яка автоматизує цю роботу, повинна знаходити на PNG зображенні особові рахунки та показання навпроти, переводити їх у тип STRING.

![image](https://user-images.githubusercontent.com/125378481/219016471-7084a64e-de10-4251-bf84-167315c44124.png)
![image](https://user-images.githubusercontent.com/125378481/219016485-7ef3e06e-ab14-44bb-8c9b-396c2eeaa66f.png)

Для реалізації цього завдання я навчив чотири нейромережі та розмітив навчальні вибірки до них. Перша нейромережа була навчена знаходити особові рахунки, друга - показання лічильника, потім знайдені області нарізалися алгоритмом на одиниці (15-76-100 -> 1,5,-,7,6,-,1,0,0) (017027 -> 0, 1, 7, 0, 2, 7), після чого третя та четверта нейромережі вгадували цифри.
У четвертій, що працювала з людським почерком, мені вдалося досягти 80% успішного вгадування цифри, і відсоток зростав разом зі збільшенням навчальної вибірки.

Кожна з цих чотирьох нейромереж це класична нейромережа з двома прихованими шарами. Структуру нейромережі я взяв із книги Тарика Рашида «Створюємо нейронну мережу». Переписав з Python на C# і додав другий прихований шар, побудовано на пакеті NuGet MathNet.Numerics (матриці, лінійна алгебра).

![image](https://user-images.githubusercontent.com/125378481/219018896-7e73204d-38ba-4eef-970d-abb365339336.png)

Основою проекту є мій незакінчений проект шахового зору. З нього я взяв функцію Int_palette, що перетворює зображення зі стандартною палітрою пікселів (1 млн кольорів) в інтовий масив з дуже невеликим діапазоном кольорів (8,27,64),

![image](https://user-images.githubusercontent.com/125378481/219019334-a33b68d1-b9fc-4e17-b2be-79597376fc2e.png)

та функцію Perimeter, що дозволяє знаходити на зображенні окремі елементи.

![image](https://user-images.githubusercontent.com/125378481/219019477-e6679487-33fc-4e03-96e6-6202650d2a93.png)
![image](https://user-images.githubusercontent.com/125378481/219019500-c8fbb15e-a4ad-4e62-859d-3259187c8789.png)
![image](https://user-images.githubusercontent.com/125378481/219019521-ad5de4cb-317a-4353-83db-3f37d74cab01.png)
![image](https://user-images.githubusercontent.com/125378481/219019530-9d06223c-56ff-4a44-8e78-23a36703da8a.png)

Процес розробки був складним, багато ідей не пройшли перевірки практикою, щоб сильно не розтягувати цей документ, я опущу невдалі.
Для простоти розуміння, як працює програма, послідовно опишу, як працюють її розібрані частини. Програма умовно поділена на закоментовані сегменти, до яких я звертаюся для збільшення виборок. (Нейромережі навчені працювати тільки з одним почерком і манерою написання показань).

Код програми знаходиться у файлі OperatorVision/OperatorVision/Form1.cs

async void button1_Click(object sender, EventArgs e)

Ця кнопка відкриває вікно openFileDialog1, де ми повинні вибрати Png зображення зі сторінками показань контролерів. Сканер на моїй роботі здатен сам перегортати сторінки і витрачав на одну менше секунди. Автоматичне перегортання рухало пачку аркушів ліворуч-праворуч у межах 5 мм, через це межі порожнього простору на кожній просканованій сторінці різні. Перший сегмент приводить усі зображення до єдинопорядку.

     //////cut_image empty_lines
     
![image](https://user-images.githubusercontent.com/125378481/219022413-4b0d08a9-9de4-4d57-a6f5-1ec3027bb11c.png)
![image](https://user-images.githubusercontent.com/125378481/219022438-63729d7f-66f9-4eb7-b357-29469049de98.png)

(На першому зображенні таблиця починається з 70того пікселя, на другому з 140го)

![image](https://user-images.githubusercontent.com/125378481/219023258-73d4035e-fe95-4422-9629-ac8364dce0fa.png)
![image](https://user-images.githubusercontent.com/125378481/219023275-e5f89074-4161-4e26-829d-2e3a58e9823a.png)

(Ті ж сторінки, але обрізані)

Усі зображення, вибрані в openFileDialog1, перезаписуються.
функція Int_palette перетворює зображення на інтовий масив, де цифри від 1 до 8 означають певний діапазон кольорів.
(1 – білий, 4 – червоний, 8 – чорний)
функція cut_empty вирізає з масиву pixel_array всі порожні (> 90% білого кольору) рядки та стовпці до першого неспрацьовування цієї умови, перезаписує масив з новою кількістю рядків та стовпців. Потім зображення зберігається назад у PNG. Невелика пауза await Task.Delay() запускає смітник, який вивільняє зайняту оперативну пам'ять.

     //////cut_image

Тепер, коли сторінки приведені до єдиного порядку, координати по горизонталі особових рахунків та показань постійні. Сторінки нарізаються на смуги з областями, де знаходяться особові рахунки та можливі показання. Іноді при перегортанні пачка листів оберталася на пару градусів, через що початок варіюється в межах 5-10 пікселів, тому нарізка здійснюється з невеликим запасом.

![image](https://user-images.githubusercontent.com/125378481/219024477-6561d3ec-e773-4f4a-8889-407defcd3bb3.png)
![image](https://user-images.githubusercontent.com/125378481/219024517-7709d5b5-75dd-4244-bf5e-0bb00819a46f.png)

     //////cut_lines
смуги особових рахунків та показань нарізаються функцією cut_lines на рядки. Висота рядків для смуг показань трохи вища, ніж для особових рахунків, що дозволяє вилучити більшу частину непотрібних рядків.
 
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

     //////Розмітка.
Після того, як були зібрані рядки, їх необхідно розмітити (які рядки є особовими рахунками або показами, а які -ні). Процес розмітки змінювався, і я не записав останній вдалий варіант. на цих гіфках ви можете подивитися, як формувалася навчальна вибірка для раннішої версії програми.

![1](https://user-images.githubusercontent.com/125378481/219029829-0a876fd9-702b-4399-81d7-4cd14826d31a.gif)

![3](https://user-images.githubusercontent.com/125378481/219032877-d5d51f32-c4f0-42d6-83da-986db7094b85.gif)

Загалом роботу нейромереж, навчених на минулих вибірках, можна назвати задовільною, і можливо з додатковими розміченими зображеннями, вони працювали б краще.
Однак останнє рішення працює бездоганно і зі своєю маленькою навчальною вибіркою.
 
 ![image](https://user-images.githubusercontent.com/125378481/219030636-847756fc-4354-4f70-9855-59f7c5dddfa8.png)
 
(кожне зображення, що не показання, змінює назву на «m»)

     //////delete quadrilaterals
Межі таблиці в нарізаних смугах не дають працювати алгоритму, що поділяє показання на одиниці, функція delete_quadrilaterals їх видаляє. (Стовпці та рядки, що містять рівні чорні смуги завдовжки 15-20 % від висоти-довжини, вилучаються, зображення перезаписується.

![image](https://user-images.githubusercontent.com/125378481/219032427-29b0fa74-d032-4bcc-abc8-e100c804cf66.png)

![image](https://user-images.githubusercontent.com/125378481/219032458-e9b6c94d-41cd-4f93-ad4d-1573f5ab1cdd.png)

![image](https://user-images.githubusercontent.com/125378481/219032481-f76dd5ef-f3cd-4695-a9a1-3a484e9f0349.png)

![image](https://user-images.githubusercontent.com/125378481/219032500-440b2e84-706c-4141-a69e-a3af352dab53.png)

![image](https://user-images.githubusercontent.com/125378481/219032521-7b7cb99b-4d6f-46d4-b57d-3ff74111edbf.png)

![image](https://user-images.githubusercontent.com/125378481/219032560-6e36c5d0-7624-4ab5-a2a2-86e0a56be6fe.png)

Через цю функцію цифри сильно втрачають в якості, що також є причиною падіння ефективності четвертої нейромережі. У ранніх версіях програми у мене були спроби навчити нейромережу ділити показання на цифри, але з невеликою навчальною вибіркою її результати були посередніми.

     //////txt
Розмічені зображення перезаписуються в txt файл. Він відкривається менше секунди, коли на тисячі зображень йдуть хвилини.
На пошуки оптимального стану нейромережі витрачається багато спроб, навчальну вибірку в оперативну пам'ять потрібно отримувати швидко.

Для економії ресурсів і скорочення часу необхідного для навчання зображення стискаються, при цьому людський почерк стає важко зрозумілим, тому зберігаємо і чисту копію, що не стискається. Функція cut_jpg рівномірно викидає до заданих параметрів стовпці та рядки, перезаписує масив.

![image](https://user-images.githubusercontent.com/125378481/219036400-06ea90d6-1f58-4af3-b035-0a739ec1cb61.png)

![image](https://user-images.githubusercontent.com/125378481/219036480-dc24fc6b-8000-4a94-89c0-59da28f3b7a5.png)

     //////first part
це канва для навчання нейромережі, що відповідає, чи є у вставленому в неї рядку особовий рахунок чи ні.
Навчання інших нейромереж проходило схожим чином, я не описуватиму кожну, опишу тільки першу.

Функція read_input записує навчальну вибірку в інтовий масив, змінна num – кількість всіх зображень, записаних у цей масив. Перевірка нейромережі на навчання буде здійснюватися на останніх 200 зображень.
  
//for (int d1 = 0; d1 < 15; d1++)

У цьому циклі підбирається зручний коефіцієнт навчання delta, гра з іншими параметрами нейромережі не приносила жодних результатів.
У нейромережу Intelect2[0] вставляється одномірний double масив, перетворений з двовимірного (100х10), у двох прихованих шарах відбувається процес «мислення», який формує double масив з одним елементом, де значення цього елемента, що прагне до 0,01 каже, що у вставленому рядку особового рахунку немає, а значення,
що прагне до 0,99, каже, що є.

//for (int c = 0; c < 10000; c++)

Навчальна вибірка порівняно невелика, кількість її елементів недостатньо для повноцінного навчання, але ці елементи можна використовувати кілька разів.
З функцією Random() нейромережа навчається краще, ніж циклом, який послідовно повторює всі елементи навчальної вибірки.

// for (int c = num; c <num + 200; c++)

На розмічених 200 зображеннях, що не брали участь у навчанні, підраховується ефективність нейромережі.
Потім результати та ваги нейронів мережі записуються в txt файли.

![image](https://user-images.githubusercontent.com/125378481/219041985-4c884467-cc30-4d1e-b9db-66c2578d61f7.png)

(файл із збереженою навченою нейромережею)

     //////mechanic digits
навчання нейромережі, що вгадує друковані цифри, які нарізані з особових рахунків функцією Perimeter.
Я планував визначати друковані цифри перевіркою на відповідність із заздалегідь підготовленими кальками цифр, але цей спосіб давав збої. Просканована друкована цифра часто є унікальною (скан у низькому розширенні), тому довелося використовувати нейромережі.

![image](https://user-images.githubusercontent.com/125378481/219044207-89665242-5542-4893-ae74-95d56daa6454.png)
![image](https://user-images.githubusercontent.com/125378481/219044226-3d912b13-fcd2-420e-b1de-e80db5bb9910.png)
![image](https://user-images.githubusercontent.com/125378481/219044246-48547c7c-b091-4903-a116-53383d4bf49e.png)

![image](https://user-images.githubusercontent.com/125378481/219044268-d342c1c2-b473-4b74-8476-89179dfa760e.png)
![image](https://user-images.githubusercontent.com/125378481/219044292-7148de3b-9662-4b77-a612-2403ef586abc.png)
![image](https://user-images.githubusercontent.com/125378481/219044315-1163d6e7-bcb4-4dd1-9f89-9eb7aac513bd.png)

![image](https://user-images.githubusercontent.com/125378481/219044344-346d42ae-da48-4e03-8b84-61a675fc936a.png)
![image](https://user-images.githubusercontent.com/125378481/219044356-2201f83c-9632-4136-ab8f-c136368adb62.png)
![image](https://user-images.githubusercontent.com/125378481/219044376-d7fe30d5-26ae-4da5-9ade-8c9bc1f2fcc9.png)

Сегмент здебільшого ідентичний third part , де нейромережа вгадує цифри, написані рукою.

     //////second part
Навчання нейромережі, що відповідає, чи є у вставленему у неї рядку показання чи ні. Нейромережа має трохи меншу ефективність (96-98%), ніж та, що працює з особовими рахунками (99%+), але це не заважає об'єднаній програмі. Особові рахунки і показання паралельні, помилкові спрацьовування відсіюються, тому що не мають пари особового рахунку.

![image](https://user-images.githubusercontent.com/125378481/219045492-1872983d-ec93-4281-8912-58b1c974ed9d.png)
(нейромережа помилково визначила, що цей рядок - показання)

     //////third part
Навчання нейромережі, що вгадує рукописні цифри, які нарізані з рядків-показів функцією Perimeter.
Спочатку я використав MNIST датасет (60к розмічених цифр), але на цифрах, написаних контролером, нейромережа показала жахливі результати (<50%, хоча на MNIST перевірочному датасеті відсоток успішних вгадувань був близько 98). Довелося проводити розмітку цифр самостійно. З двома тисячами розмічених цифр, я досяг 80 відсотків успішних вгадувань, і можливо зі збільшенням навчальної вибірки відсоток досяг би планки 90-95. Однак я впевнений, що це тупикова гілка розвитку. Область, де пишуть показяння, доцільніше замінити поштовим індексом.

![image](https://user-images.githubusercontent.com/125378481/219046882-bfcab13f-1cdc-424c-a768-2e7350994698.png)

     //////united
У мене не дуже багато просканованних сторінок контролерів, з усіх були сформовані навчальні вибірки, крім останніх п'ятьох. На них здійснюється перевірка об'єднаної програми.(відео за посиланням
https://drive.google.com/file/d/1CkTENVJSMnnw9AY8-0b71aGU57q4PX2-/view?usp=sharing)

     //////button2_Click(object sender, EventArgs e) (можливе майбутнє)
Щоб трохи полегшити собі процес формування навчальних вибірок, я написав невеликий клікер, що заощаджує мені 5-10 секунд для кожного розміченого зображення (ви вже бачили його роботу у гіфках).
Дещо зміненим клікером можна було користуватися для внесення до бази показань (API у програмі-базі не було).
До показань справа не дійшла, але я вводив подібним чином кілька тисяч телефонів абонентів. (відео за посиланням https://drive.google.com/file/d/1wuzVtq7a08TJmGSePOl2WcHsZJFgsQSP/view?usp=sharing)
 
 

