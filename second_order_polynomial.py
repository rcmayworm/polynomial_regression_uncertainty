import numpy as np
from uncertainty import Uncertainty
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

#choice of regression
step = 'linear'
step = 'polynomial'

#open data
data = np.genfromtxt("data.txt", delimiter=";")
x = data[:,0]
y = data[:,1]

#call methods
uncertainty = Uncertainty()
linear_regression = uncertainty.uncertainty_linear_regression(x,y)
polynomial_regression = uncertainty.uncertainty_polynomial_regression(x,y)

if step == 'linear':
    y_predict = linear_regression[3].reshape(-1, 1).flatten()
    U = linear_regression[7]
    R2 = float('{0:.4f}'.format(linear_regression[2]))
    AC = float('{0:.2f}'.format(linear_regression[0]))
    IC = float('{0:.2f}'.format(linear_regression[1]))
    Veff = linear_regression[8]
    ResidualErrorValues = linear_regression[4]
    print(f'sy: {linear_regression[10]**2}')

    y_minus = (y_predict - U).reshape(-1, 1).flatten()
    y_plus = (y_predict + U).reshape(-1, 1).flatten()
    legend = 'R² = ' + str(R2) + '\nP = ' + str(IC) + ' + (' + str(AC) + ')mV' + '\nU = ' + str(
        float('{0:.2f}'.format(max(U)))) + 'mm' + ' k = ' + str(
        float('{0:.2f}'.format(Veff)))
else:
    y_predict = np.array(polynomial_regression[4])
    U = polynomial_regression[6]
    R2 = float('{0:.4f}'.format(polynomial_regression[3]))
    b_0 = float('{0:.2f}'.format(polynomial_regression[0]))
    b_1 = float('{0:.2f}'.format(polynomial_regression[1]))
    b_2 = float('{0:.2f}'.format(polynomial_regression[2]))
    Veff = polynomial_regression[7]
    ResidualErrorValues = polynomial_regression[5]
    print(f'sy: {polynomial_regression[9]**2}')

    y_minus = (y_predict - U)
    y_plus = (y_predict + U)
    legend = 'R² = ' + str(R2) + '\nP = ' + str(b_0) + ' + (' + str(b_1) + ')mV' + ' + (' + str(b_2) + ')mV²' + '\nU = ' + str(
        float('{0:.2f}'.format(max(U)))) + 'mm' + ' k = ' + str(
        float('{0:.2f}'.format(Veff)))

#plot graph
plt.figure(1)
plt.title('Calibration')
plt.plot(x, y,  'ro', label='Measurements')
plt.plot(x, y_predict,  '--', label='Polynomial regression')
plt.fill_between(x, y_minus, y_plus, color='gray', alpha=0.2)
plt.legend(title=legend, loc='best', fontsize='x-small', title_fontsize=8)
plt.xlabel('Reading [mV]')
plt.ylabel('position [mm]')
plt.show()

N = len(ResidualErrorValues)
zero = [0] * int(N+1)
x_axis = list(range(1,(N+1)))

plt.figure(2)
plt.title('Residue')
plt.errorbar(x_axis, ResidualErrorValues, yerr= U,  fmt='or', ecolor='black', elinewidth=1, capsize=5)
plt.plot(zero,  '--')
plt.xlabel('Measure')
plt.ylabel('Error [mm]')
plt.show()
