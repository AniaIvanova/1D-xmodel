#data from CON1D16.03.01
#javnaja sxema, 3xtoczeczny shablon, priamougolnye koordinaty
from math import *

life=25000##280            #число шагов по времени

half_plast=0.15      #polutolschina plastiny
v=0.0167             #skorost razlivki 1m/min

mju=243000.0   #   168000.0     #teplota kristalizacii  168kJ/kg
qalong=[]      #   q - mold heat transfer along strand surface kW/m2
liq=[]              #polozenie liquidus
tmp_liq=[] #vremia dla liq
sol=[]              #polozenie solidus
tmp_sol=[] #vremia dla sol
Tl=1796             #temperatura liquidus
Ts=1760             #temperatura solidus
N=100                #czislo szagow po prostranstwu   (22 ne rabotaet xz why)
delta_x=half_plast/N
tmpr=0              #temperatura
t=[]                #temperatura tabliza

otmeniska=[]        #distance from meniskus
otmeniska_l=[]      #dla grafika liquidusa, shob ne risovalsia nol posle lfd
otmeniska_s=[]      #to ze dla grafika solidusa

delta_tau=0.075     # шаг по времени в секундах
# условие устойчивости delta_tau <= c_min * ro_min * delta_x**2 / (2 * lam_max)
# proveriaetsa nize, posle zadania tfh tablic

#lambda: heat_transfer depends on temperature by table_lambda 
#c: table_heatcapacity 
#ro: density depends on temperature by table_ro
table_heatcapacity=[[975,660],[1031,635],[1032,610],[1079,592],[1727,696],[1757,720],[1758,743],[1773,748],[1797,824],[1873,846]]
table_ro=[[975,7700],[1079,7500],[1727,7330],[1757,7280],[1758,7265],[1773,7230],[1797,7020],[1873,6600]]  # плотность
table_lambda=[[975,33],[1079,28],[1757,33],[1797,34.8],[1873,34.8]]    # теплопроводность

#Heat Flux by time
def hfq_old(tau):
    qq = 250000+640000*exp((-0.4)*tau*v)-2000.0*tau*v
    return qq
    
#Heat Flux by time
def hfq(tau):
    qq = 1000000*6.36/((tau + 4.5)**(1/2))
    #qq = 250000+2560000*exp((-1.2)*tau*v)-2000.0*tau*v
    #qq = 1000000*(0.11 + 3.36/((tau + 1.032)**(1/2)))
    return qq    
    
def q_standard(tau):
    qst = 1000000*6.36/((tau + 1.032)**(1/2))  #Zappulla. Effect of Grade on Thermal–Mechanical Behavior
    return qst
       

# print ('Найти максимум вторых значений таблицы:')
def max_table_by2(table):
    max_by2=table[0][1]
    for i in range(1,len(table)):
        if table[i][1]>max_by2 :
            max_by2=table[i][1]
    return max_by2

# print ('Найти минимум вторых значений таблицы:')
def min_table_by2(table):
    min_by2=table[0][1]
    for i in range(1,len(table)):
        if table[i][1]<min_by2 :
            min_by2=table[i][1]
    return min_by2



def linealinterpol(x1,x2,y1,y2,x):
    return ((y2-y1)/(x2-x1))*(x-x1)+y1    

#linejnaja interpolacja po dwum sosednim znaczeniam iz tablicy
def table_interpol(tt,table):
    i=0
    while tt>table[i][0]:
        i=i+1
    x1=table[i-1][0]
    y1=table[i-1][1]
    x2=table[i][0]
    y2=table[i][1]
    return linealinterpol(x1,x2,y1,y2,tt)

def c(tmpr):
    #c_ef = table_interpol(tmpr,table_heatcapacity)
    c_ef = 680 # from CON1D16.03.01 .inp file
    #dla razmaz dobavliaem effektivnuju sostavliajusju
    if tmpr>=Ts and tmpr<=Tl:
        c_ef = c_ef+mju/(Tl-Ts)
    return c_ef

def lam(tmpr):
    return 31 # from experiments
    # old return table_interpol(tmpr,table_lambda)

def ro(tmpr):
    return 7000 # from CON1D16.03.01 .inp file
    # old return table_interpol(tmpr,table_ro)

# proverka условие устойчивости delta_tau <= c_min * ro_min * delta_x**2 / (2 * lam_max)
#delta_tau_max = min_table_by2(table_heatcapacity) * min_table_by2(table_ro) * delta_x**2 / (2 * max_table_by2(table_lambda))
delta_tau_max = c(1000) * ro(1000) * delta_x**2 / (2 * lam(1000))
#print ('min_table_by2(table_heatcapacity) = ', min_table_by2(table_heatcapacity))
print ('delta_tau_max < ', delta_tau_max)
if delta_tau>delta_tau_max :
    print ('Achtung!!! Шаг по времени delta_tau=', delta_tau,' больше допустимого по условию устойчивости: delta_tau_max=', delta_tau_max)

print('0 : hfq = ', hfq(0/v))
print('0.1 : hfq = ', hfq(0.1/v))
print('0.2 : hfq = ', hfq(0.2/v))
print('0.26 : hfq = ', hfq(0.26/v))
print('0.3 : hfq = ', hfq(0.3/v))
print('0.5 : hfq = ', hfq(0.5/v))
print('0.7 : hfq = ', hfq(0.7/v))
print('1 : hfq = ', hfq(1/v))
print('1.5 : hfq = ', hfq(1.5/v))
print('2 : hfq = ', hfq(2/v))
print('3 : hfq = ', hfq(3/v))
print('10 : hfq = ', hfq(10/v))


# решаем уравнение теплопроводности



# задаем начальные условия
#naczalnaja temperatura
#table_initial_T=[[0,1806],[0.05,1806],[0.07,1806],[0.98*half_plast,1806],[half_plast,1806]] # from CON1D16.03.01
#table_initial_T=[[0,1823],[0.05,1820],[0.07,1812],[0.98*half_plast,1812],[half_plast,1812]] # bylo [0.095,Ts],[0.1,Ts-1]

x = [i*delta_x for i in range(0,N+1)] # узлы сетки, poslednij el-t v in range ne vkluchaetsa
#def f(x):
#    return 50000000*(x**2-0.06*x+0.0008)*(x**2-0.14*x+0.0048)+1500
def T0(x):
    #return table_interpol(x,table_initial_T)
    #return ((Ts+Tl)/2 -1823)*(x**28)/(half_plast**28) + 1823 #parabola
    return 1806.15

t=[[T0(x[i]) for i in range(0,N+1)]]

#dla razmaz
#po naczalnomu raspredeleniu t-ry nahodim naczlnye liq i sol
if t[0][N]>=Ts: ssol = half_plast
for i in range(0,N):
    if t[0][i]>Ts and t[0][i+1]<=Ts:
        ssol = linealinterpol(t[0][i], t[0][i+1], x[i], x[i+1], Ts)
if t[0][N]>=Tl: lliq = half_plast
for i in range(0,N):
    if t[0][i]>Tl and t[0][i+1]<=Tl:
        lliq = linealinterpol(t[0][i], t[0][i+1], x[i], x[i+1], Tl)
#print('lliq[0]=',lliq, ' :  ssol[0]=',ssol)
liq.append(half_plast - lliq)
sol.append(half_plast - ssol)
tmp_liq.append(0)
tmp_sol.append(0)
otmeniska.append(0)
otmeniska_l.append(0) #for graphic liqsol
otmeniska_s.append(0) #for graphic liqsol
#qalong.append(1500000)#for grafic of heat flux from surface

#dla Stef nahodim nacz ksi gde average (Ts+Tl)/2
if t[0][N]>=(Ts+Tl)/2: ksi = half_plast
else: ksi=linealinterpol(t[0][i], t[0][i+1], x[i], x[i+1], (Ts+Tl)/2)
#print('ksi[0]=',ksi)

#javnaja sxema, 3xtoczeczny shablon, priamougolnye koordinaty
def next_t(t_im1,t_i,t_ip1):
    # bez istocnika
    #r=(delta_tau/(c(t_i)*ro(t_i)*(delta_x**2)))*((lam(t_i)-lam(t_im1))*(t_i-t_im1)+lam(t_i)*(t_ip1-2*t_i+t_im1))+t_i

    #s istochnikom
    u = 0               #isto4nik / stok  tepla
    r=delta_tau*((lam(t_im1)*t_im1 - lam(t_im1)*t_i - lam(t_i)*t_i + lam(t_i)*t_ip1)/(delta_x**2) + u)/(c(t_i)*ro(t_i)) + t_i
    return r

otmenisk = 0
tempdisl = 0.0 #for graphic liqsol
tempdiss = 0.0 #for graphic liqsol
#ras4et vseh 6agov po vremeni 
for j in range(0,life+1):
    otmenisk = otmenisk + delta_tau*v
    otmeniska.append(otmenisk)
    t.append([0 for i in range(0,N+1)])
    t[j+1][0]=t[j][0]   
    t[j+1][N]=t[j][N]
    #peres4et po trehto4e4nomu shablonu
    for i in range(1,N): t[j+1][i]=next_t(t[j][i-1],t[j][i],t[j][i+1]) 


    #===============================================================================================
    #granicnye uslovia 
    #gran usl II-go roda, na osi teplovoj potok =0
    t[j+1][0] = t[j+1][1]   

    #gran usl II-go roda, na poverhnosti teplovoj potok q = hfq
    q = hfq(otmenisk/v)
    
    
    #table_mfh=[[0.0, 1500000],[0.02, 1700000],[0.15, 1470000],[0.27, 1200000],[0.53, 1010000],[0.67, 1000000], [32.0, 250000]] 
    #q = table_interpol(otmenisk,table_mfh)
    qalong.append(q)
    #print ('q = ', q)
    t[j+1][N]= -q*delta_x/lam(t[j][N]) + t[j+1][N-1]
 
    
    #liq sol : na etom ze shage vremeni najdem liq i sol
    for i in range(0,N):
        if t[j+1][i]>Tl and t[j+1][i+1]<=Tl:
            lliq = linealinterpol(t[j+1][i], t[j+1][i+1], x[i], x[i+1], Tl)
            #print('i=',i)
        if t[j+1][i]>Ts and t[j+1][i+1]<=Ts:
            ssol = linealinterpol(t[j+1][i], t[j+1][i+1], x[i], x[i+1], Ts)
    if t[j][0]>Tl and t[j+1][0]<=Tl: lliq = 0
    if t[j][0]>Ts and t[j+1][0]<=Ts: ssol = 0
    if t[j+1][N]>=Ts: ssol = half_plast
    liq.append(half_plast - lliq)
    
    prelast_element_liq = liq[-2]
    #print('prelast_element_liq = ',prelast_element_liq)
    if lliq > 0:
        tlliq=(j+1)*delta_tau
        tempdisl = tempdisl  + delta_tau*v #for graphic liqsol 
    if (lliq == 0) and (prelast_element_liq < half_plast) :
        tempdisl = tempdisl  + delta_tau*v
        # print('prelast_element_liq = ',prelast_element_liq)
        print('tempdisl = ',tempdisl)
    otmeniska_l.append(tempdisl)           #for graphic liqsol
        
    tmp_liq.append(tlliq)
    
    
    
    sol.append(half_plast - ssol)
    prelast_element_sol = sol[-2]
    if ssol > 0:
        tssol=(j+1)*delta_tau
        tempdiss = tempdiss  + delta_tau*v #for graphic liqsol
    if (ssol == 0) and (prelast_element_sol < half_plast) :
        tempdiss = tempdiss  + delta_tau*v
       # print('prelast_element_sol = ',prelast_element_sol)
        print('tempdiss = ',tempdiss)
    otmeniska_s.append(tempdiss)           #for graphic liqsol
    tmp_sol.append(tssol)
    if j < 3 : print('j=',j,' lliq=',lliq, ' :  j=',j,' ssol=',ssol)
    
print('liqidus depth: tempdisl = ',tempdisl)  #one additional segment for  
print('solidus depth: tempdiss = ',tempdiss)  # the first ssol lliq = 0
    
from numpy import *
#from math import *   #уже было в начале, но нафигато здесь ещё раз (без него не пашет)
import math
import matplotlib.pyplot as plt

#interpolacja po stroke w tablice
def table2d_interpol(table,stroka,tmp):
    x1=int(floor(tmp))
    if x1==tmp:
        return table[stroka][x1]
    x2=x1+1
    y1=table[stroka][x1]
    y2=table[stroka][x2]
    return linealinterpol(x1,x2,y1,y2,tmp)

#t - tablica vseh wyczislenyh temperatur, g - znaczenie w toczke tmp interpolirowanoe po i-j stroke tablicy t  
def g(tmp):
    return table2d_interpol(t,i,tmp)

#stroim grafiki raspredelenia temperatury w raznye momenty vremeni
#for i in range(0,life+1): #perebirajem wse stroki tabl t - vse wyczislen sloi po wremeni
for i in range(0, life+1, 500): #perebirajem tolko kazduu 500-ju stroku tabl t
    tmp = linspace(0, len(t[i])-1, 500)  #poslednee число - kol-vo точек для построения графика 
    y=[]
    for x in tmp:
        y.append(g(x)-273.15) #g - znaczenie w toczke tmp (sm. vyshe)
    plt.plot(tmp*delta_x, y)  #plt.plot(tmp*delta_x, y, 'red')- all will be red
plt.title('Temperature profile from axis to surface') # every ?? sec
plt.xlabel('thickness, m')
plt.ylabel('Temperature, °С')
# включаем дополнительные отметки на осях
plt.minorticks_on()
plt.savefig('name_of_plot.jpg', dpi=300) #dpi больше - больше рисунок w jpg
plt.show()   #pokazat grafik w konsoli


#stroim grafik t-ry centra i poverhnosti
#stroim grafik na intervale [0, life*delta_tau]
#massiv argumentov tmp
tmp = linspace(0, life*delta_tau, life+1)
#massiv znaczenij y
mark = ['.', 'v', '*', '^', 'p'] #tipy markerov  
bce = math.floor(N/5 + 0.5) #blizajshee celoe
for j in range(0, N, bce): 
    y=[]
    for i in range(0,life+1):
        y.append(t[i][j]-273.15)
    my_string = str(j*delta_x*1000)
    plt.plot(tmp, y, marker = mark[j//bce], markevery=1800, label=my_string+'mm')

y=[]
for i in range(0,life+1):
    #print(i)
    y.append(t[i][N]-273.15)
#teper mozno stroit grafik s massivom argumentov i znaczenij
plt.plot(tmp, y, 'steelblue', label='surface')
plt.title('Temperature of surface, layers and center')  #every ?? mm
plt.xlabel('time, sec')
plt.ylabel('Temperature, °С')
# включаем дополнительные отметки на осях
plt.minorticks_on()
plt.legend()
    
#sohraniaem kartinku
plt.savefig('graf_t_poverhn.jpg', dpi=300) #dpi больше - больше рисунок w jpg
plt.show()


import numpy as np    
#stroim grafik mold heat flux
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twiny()

X = linspace(0, life*delta_tau, life+1)
#Y = 225000+510000*exp((-0.4)*X*v)-3500*X*v 
Y = hfq(X)/1000000
Y2 = q_standard(X)/1000000
Y3 = hfq_old(X)/1000000

ax1.plot(X, Y2, 'orange', label='Zappulla')
ax1.plot(X, Y, 'navy', linestyle='-.', label='Heat Flux now')
ax1.plot(X, Y3, 'green', linestyle=':', label='old Heat Flux') 
#line styles: '--' dashed, '-.' dash-dot, 'o' line with circular markers
ax1.set_xlabel(r"time, sec")
ax1.legend()

'''
new_tick_locations = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])

def tick_function(X):
    V = X*v
    return ["%.3f" % z for z in V]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"distance from meniscus, m")
'''

# включаем основную сетку
plt.grid(which='major')
plt.ylabel('Heat Flux, MW/m2')
plt.title('HEAT FLUX of surface')
plt.show()

#stroim grafik na intervale [0, (life+1)*delta_tau]
#massiv argumentov tmp
  #tmp = linspace(0, (life+1)*delta_tau, life+1)
#stroim grafik po massivu argumentov i massivu znaczenij
  #plt.plot(otmeniska, qalong, 'orangered')
  #plt.title('Surface heat flux')
  #plt.xlabel('Distance below meniscus, m')
  #plt.ylabel('Heat Flux, W/m2')
  #plt.savefig('heatflux.jpg', dpi=300) #dpi больше - больше рисунок w jpg
  #plt.show()


#stroim grafiki liq i sol
#stroim grafik na intervale [0, (life+1)*delta_tau]
#massiv argumentov tmp
tmp = linspace(0, (life+1)*delta_tau, life+2)
#stroim grafik po massivu argumentov i massivu znaczenij
plt.plot(otmeniska_l, liq, 'orangered', label='liqidus')
plt.plot(otmeniska_s, sol, 'steelblue', label='solidus')
plt.title('Liquidus and Solidus Isoterms')
plt.xlabel('Distance below meniscus, m')
plt.ylabel('Distance from strand surface, m')
# включаем дополнительные отметки на осях
plt.minorticks_on()
plt.legend()
plt.savefig('liqsol.png', dpi=300) #dpi больше - больше рисунок w jpg
plt.show()


#export dannyh w excel
import xlsxwriter 
# Workbook() takes one, non-optional, argument which is the filename that we want to create 
workbook = xlsxwriter.Workbook('liqsol.xlsx') 
  
# The workbook object is then used to add new worksheet via the add_worksheet() method 
worksheet = workbook.add_worksheet() 
  
# Use the worksheet object to write data via the write() method 
worksheet.write('A1', 'Hello..') #A kolonka, 1 - nomer stroki

#Zapis w excel liquidusa i solidusa
# First and second numbers are row/column notation.
worksheet.write(0, 0, 'time, s')
worksheet.write(0, 1, 'distance, m') 
worksheet.write(0, 2, 'liquidus, m')
worksheet.write(0, 3, 'solidus, m')
every_n = 1 #write in file only every n-th result
for i in range(0,life-1):
    if i%every_n < 0.01 :   #reminder of division
        line_number = i//every_n  #integer part of division
        worksheet.write(line_number+1, 0, tmp[i])
        tdistance = tmp[i]*v
        worksheet.write(line_number+1, 1, tdistance)
        if  (liq[i-1]<half_plast) :     #stop writing after liq is half_plast
            worksheet.write(line_number+1, 2, liq[i])
            if  liq[i] > (half_plast-0.001) :
                print('i = ', i, '  :  liq[i] = ', liq[i], 'liquid phase depth = ', (i)*delta_tau*v)                 
        if (sol[i-1]<half_plast) : worksheet.write(line_number+1, 3, sol[i])


# Insert an image.
worksheet.insert_image('E1', 'liqsol.jpg') #E - kolonka, 1 - nomer stroki

# Finally, close the Excel file via the close() method 
workbook.close() 

#Excel file temperatury poverchnosti, centra i raznych sloev
workbook = xlsxwriter.Workbook('Tsurf+.xlsx') 
worksheet = workbook.add_worksheet() 
worksheet.write(0, 0, 'time, s') 
worksheet.write(0, 1, 'Tsurf, K')
worksheet.write(0, 2, 'Tcenter, K')
for i in range(1,life):
    worksheet.write(i, 0, tmp[i])
    j1=0
    for j in range(0, N+1, math.floor(N/5 + 0.5)): #blizajshee celoe
        j1=j1+1
        worksheet.write(i, j1, t[i][j]-273.15)

workbook.close()

#print(otmeniska_l)
exit_condition=input("press Enter or Close to exit")
