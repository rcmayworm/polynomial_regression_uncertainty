from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import t
import numpy as np
import math
from math import sqrt

class Uncertainty():
    def uncertainty_linear_regression(self, Axis_X, Axis_Y):
        # list to array
        Axis_X = np.array(Axis_X).reshape(-1, 1)
        Axis_Y = np.array(Axis_Y).reshape(-1, 1)

        reg = LinearRegression()
        reg.fit(Axis_X, Axis_Y)
        Axis_Y_predicted = reg.predict(Axis_X) #predict values
        R2 = reg.score(Axis_X, Axis_Y) # R2 value
        linreg = LinearRegression().fit(Axis_X, Axis_Y)
        AC = linreg.coef_[0][0]  # angular coefficient
        IC = linreg.intercept_[0]

        N = len(Axis_X)
        Residue = (Axis_Y - Axis_Y_predicted)
        S = math.sqrt((np.sum(np.square(Axis_Y - Axis_Y_predicted))) / (N - 2))
        X_Mean = np.mean(Axis_X)
        X_Sum = (np.sum(Axis_X))
        X_Square = (np.sum(np.square(Axis_X)))
        X_Square_X_Mean = (np.sum(np.square(Axis_X - X_Mean)))
        u_IC = S * math.sqrt((X_Square) / (N * X_Square_X_Mean))
        u_AC = S / math.sqrt(X_Square_X_Mean)
        r = -X_Sum / math.sqrt(N * X_Square)
        C_IC = 1
        DegreesFreedom = N - 2
        Veff = (t.ppf(1.0 - 0.025, DegreesFreedom)).tolist()

        U = []
        for i in range(len(Axis_X)):
            C_AC = Axis_X[i]
            u = math.sqrt(((u_IC * C_IC) ** 2) + ((u_AC * C_AC) ** 2) + 2 * u_IC * C_IC * u_AC * C_AC * r)
            U.append(u * Veff)

        return (AC, IC, R2, Axis_Y_predicted, Residue, u_AC, u_IC, U, Veff, DegreesFreedom, S)

    def uncertainty_polynomial_regression(self, Axis_X, Axis_Y):
        Axis_X = np.array(Axis_X).reshape(-1, 1)
        Axis_Y = np.array(Axis_Y).reshape(-1, 1)

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(Axis_X)
        pilreg = LinearRegression()
        pilreg.fit(x_poly, Axis_Y)
        Axis_X = Axis_X.flatten()
        Axis_Y = Axis_Y.flatten()
        N = len(Axis_X)

        def f(x, b0, b1, b2):
            return b0 + (b1 * x) + (b2 * x ** 2)

        parameters, cov = curve_fit(f, Axis_X, Axis_Y)
        Axis_Y_predicted = pilreg.predict(x_poly).flatten()
        Residue = (Axis_Y.flatten() - Axis_Y_predicted)

        #coeficients = pilreg.coef_.tolist()[0]
        a = parameters[0]
        b = parameters[1]
        c = parameters[2]
        R2 = r2_score(Axis_Y_predicted, Axis_Y)
        sy = math.sqrt((np.sum(np.square(Axis_Y - Axis_Y_predicted))) / (N - 3)) #sy = σ
        #sa and sab [...] are already squared
        sa = cov[0, 0]
        sb = cov[1, 1]
        sc = cov[2, 2]
        sab = cov[0, 1]
        sac = cov[0, 2]
        sbc = cov[1, 2]

        u_final = []
        for x in Axis_X:
            dfdy = b+2*c*x
            dfda = 1
            dfdb = x
            dfdc = x**2
            u = sqrt((sy)**2 + (dfda**2)*sa + (dfdb**2)*sb + (dfdc**2)*sc) #sy não estava ao quadrado
            u_final.append(sqrt((u**2 + 2*dfda*dfdb*sab + 2*dfda*dfdc*sac + 2*dfdb*dfdc*sbc)))

        DegreesFreedom = N-3
        Veff = t.ppf(1.0 - 0.025, DegreesFreedom)
        U = [i*Veff.item() for i in u_final]
        Axis_Y_predicted = Axis_Y_predicted.tolist()

        return (a, b, c, R2, Axis_Y_predicted, Residue, U, Veff, DegreesFreedom, sy)