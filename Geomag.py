import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#файл с исходными данными
pth='E:\\Алексей\\Курсы\\PfDS\\Geomag\\Geomag.csv'

#чтение данных из файла
with open(pth) as f:
    geomag=pd.read_csv(f,sep=';')

#оставляем только напряженность вектора магнитного поля
mag_field=pd.Series(geomag['TOT'])

#в нТл числа слишком большие, веса быстро выходят за пределы разумного, приводим к единицам поменьше
mag_field=mag_field/10000

#веса и входные параметры для обучения
n=4
w=np.zeros(n)
a=0.01
b=-0.4
epoch=100

#формируем обучающую выборку из первой половины данных
D=[]
for i in range(0,50):
    x=np.array(mag_field[i:i+n])
    y=np.array(mag_field[i+n])
    D+=[[x,y]]

#функция прогноза
def f(x):
    s = b + x @ w
    return s
#тренировка
def train():
    global w
    _w = w.copy()
    for x,y in D:
        w += a * (y - f(x))*x
    return 1

#тренируем определенное количество эпох
for i in range(epoch):
    train()
    print(w)


#получаем предсказание по модели на 1 год вперед
y_pred=[]
for i in range(n+1):
    y_pred.append(mag_field[i])

for i in range(0,len(mag_field)-n-1):
    y_pred.append(f(mag_field[i:i+n]))

#предсказание на много шагов вперед (не работает, ряд уходит далеко не в ту сторону)
y2_pred=y_pred[0:n]
for i in range(len(mag_field)-n):
    y2_pred.append(f(y2_pred[i:i+n]))

#формируем рисунок с результатами
fig=plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(geomag['DATE'], mag_field, 'g', label="geomag")
ax1.set_xlabel('Года')
ax1.set_ylabel('Напряженность магнитного поля, Тл*10^-5')
line2, = ax1.plot(geomag['DATE'], y_pred, 'r', label="forecast")
plt.legend((line1, line2), ('Исходные данные', 'Модель'))
#line3, = ax1.plot(geomag['DATE'], y2_pred, 'r', label="forecast2")
plt.show()

#смотрим какие отклонения у модели от реальных данных
otkl=np.array(100*(mag_field-y_pred)/mag_field)
print('Среднее {:.2} %'.format(np.mean(otkl)))
print('Максимальное {:.2} %'.format(max(otkl)))
print('Минимальное {:.2} %'.format(min(otkl)))

#прогноз на 2021
print('Прогноз магнитного поля на 2021 год {:.7} нТл'.format(f(mag_field[len(mag_field)-n:])*10000))

