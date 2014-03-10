import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def opacity(T, rho):
	"""
	Reads opacities from a file. Finds the opacity value
	that most closely resembles the present values for
	T and R.
	Returns the opacity in units of [cm^2 g^-1].
	"""

	logT = []; logK = []
	inFile = open('opacity.txt', 'r')
	
	# Read the header file to store log(R)
	logR = np.asarray(inFile.readline().strip().split()[1:], dtype=np.float64)

	inFile.readline() # Skip the header

	# Adding log(T) and log(khappa) in separate lists
	for line in inFile:
		logT.append(line.strip().split()[0])
		logK.append(line.strip().split()[1:])

	inFile.close()
	# Converts the array to contain 64 bit floating point numbers
	logT = np.asarray(logT, dtype=np.float64)
	logK = np.asarray(logK, dtype=np.float64)

	R = rho / (T / 1e6)	# Definition of R given in the opacity file

	# Make two arrays that contain the difference in T and R from present values
	diffT = abs(10**(logT) - T)
	diffR = abs(10**(logR) - R)

	# Finds the index of the minimum difference values, so the most relevant kappa can be used.
	i = np.argmin(diffT)
	j = np.argmin(diffR)
	
	kappa = 10**(logK[i,j])
	return kappa


def energyGeneration(T, rho):
	"""
	Function to find the full energy generation per
	unit mass from the three PP chain reactions.
	"""

	ergs = 1.602e-6			# Conversion from MeV to ergs (CGS)
	NA_inv = 1./6.022e23	# Avogadro's constant inverse

	mu = 1.6605e-24		# Units of [g]		CGS

	# Abundancy of different elements
	X = 0.7				# Hydrogen (ionised)
	Y = 0.29			# Helium 4 (ionised)
	Y_3 = 1e-10			# Helium 3 (ionised)
	Z = 0.01			# Heavier elements than the above (ionised)
	Z_7Li = 1e-13		# Lithium 7 (part of Z)
	Z_7Be = 1e-13		# Beryllium 7 (part of Z)
	
	### Energy values (Q) ###
	# Energy values of nuclear reactions in PP I [MeV]
	Q_pp = (1.177 + 5.949) * ergs ; Q_3He3He = 12.86 * ergs

	# Energy values of nuclear reactions in PP II [MeV]
	Q_3He4He = 1.586 * ergs ; Q_e7Be = 0.049 * ergs ; Q_p7Li = 17.346 * ergs

	# Energy values of nuclear reactions in PP III [MeV]
	Q_p7Be= (0.137 + 8.367 + 2.995) * ergs

	### Number densities (n) ###
	n_p = X * rho / mu					# Hydrogen
	n_3He = Y_3 * rho / (3 * mu)		# Helium 3
	n_4He = Y * rho / (4 * mu)			# Helium 4
	n_7Be = Z_7Be * rho / (7 * mu)		# Beryllium 7
	n_e = n_p + 2 * n_4He				# Electron
	n_7Li = Z_7Li * rho / (7 * mu)		# Lithium 7

	### Reaction rates (lambda) ### Units of reactions per second per cubic cm. [cm^3 s^-1]
	T9 = T / (1e9)

	l_pp = (4.01e-15 * T9**(-2./3) * np.exp(-3.380 * T9**(-1./3)) * (1 + 0.123 * T9**(1./3)\
			+ 1.09 * T9**(2./3) + 0.938 * T9)) * NA_inv

	l_3He3He = (6.04e10 * T9**(-2./3) * np.exp(-12.276 * T9**(-1./3))\
			* (1 + 0.034 * T9**(1./3) - 0.522 * T9**(2./3) - 0.124 * T9 +\
			0.353 * T9**(4./3) + 0.213 * T9**(5./3)))*NA_inv

	a = 1 + 0.0495 * T9 # Defined for efficiency in l_3He4He
	l_3He4He = (5.61e6 * a**(-5./6) * T9**(-2./3) * np.exp(-12.826 * a**(1./3) * T9**(-1./3)))\
			* NA_inv
	
	# Check if temperature is below 1e6 for this reaction. Special case if true.
	if T < 1e6:
		l_e7Be = (1.51e-7 / n_e) * NA_inv
	else:
		l_e7Be = (1.34e-10 * T9**(-1./2) * (1 - 0.537 * T9**(1./3) + 3.86 * T9**(2./3)\
			+ 0.0027 * T9**(-1.) * np.exp(2.515e-3 * T9**(-1.)))) * NA_inv

	a = 1 + 0.759 * T9 # Defined for efficiency in l_p7Li
	l_p7Li = (1.096e9 * T9**(-2./3) * np.exp(-8.472 * T9**(-1./3)) - 4.830e8 * a**(-5./6)\
	* T9**(-2./3) * np.exp(-8.472 * a**(1./3) * T9**(-1./3))) * NA_inv

	l_p7Be = (3.11e5 * T9**(-2./3) * np.exp(-10.262 * T9**(-1./3))) * NA_inv

	### Rates per unit mass (r) ### 
	r_pp = l_pp * (n_p * n_p) / (rho * 2)
	r_3He3He = l_3He3He * (n_3He * n_3He) / (rho * 2)
	r_3He4He = l_3He4He * (n_3He * n_4He) / rho
	r_e7Be = l_e7Be * (n_7Be * n_e) / rho
	r_p7Li = l_p7Li * (n_p * n_7Li) / rho
	r_p7Be = l_p7Be * (n_p * n_7Be) / rho

	### Energy generation per unit mass from PP I, II and III ###
	epsilon = (Q_pp * r_pp) + (Q_3He3He * r_3He3He) + (Q_3He4He * r_3He4He) + (Q_e7Be *
			r_e7Be) + (Q_p7Li * r_p7Li) + (Q_p7Be * r_p7Be)
	
	return epsilon

def getRho(T, P):
	"""
	Calculates the density at present location.
	Returns density in units of [g cm^-3].
	"""
	### Abundancy ###
	X = 0.7
	Y = 0.29
	Z = 0.01

	### Constants ###

	# Stefan-Boltzmann constant
	sigma = 5.67e-5	# Units of [erg cm^-2 s^-1 K^-4]	CGS

	# Boltzmann constant
	k = 1.38e-16			# Units of [erg K^-1]		CGS

	# Atonmic mass unit
	m_u = 1.6605e-24		# Units of [g]				CGS

	# Speed of light
	c = 3e10				# Units of [cm s^-1]		CGS

	# Radiation constant
	a = 4 * sigma / c

	# Average molecular weight
	mu = 1./(2*X + 3./4 * Y + 9.*Z/14)

	rho = mu * m_u / (k * T) * (P - a/3 * T*T*T*T)
	return rho

def drdm(r, rho):
	"""
	Calculates the right-hand side of dr/dm.
	"""
	return 1./(4 * np.pi * r * r * rho)

def dPdm(r, m):
	"""
	Calculates the right-hand side of dP/dm.
	"""
	G = 6.67e-8		# Units of [cm^3 g^-1 s^-2]		CGS
	return - G * m / (4 * np.pi * r * r * r * r)

def dLdm(T, rho):
	"""
	Calculates the right-hand side of dL/dm.
	"""
	return energyGeneration(T, rho)

def dTdm(T, L, r, kappa):
	"""
	Calculates the right-hand side of dT/dm.
	"""
	sigma = 5.67e-5			# Units of [erg cm^-2 s^-1 K^-4]	CGS

	return -3 * kappa * L / (256 * np.pi*np.pi * sigma * r*r*r*r * T*T*T)


def integration():
	"""
	Function that integrates the equation governing
	the internal structure of the radiative core
	of the Sun.
	"""
	
	L0 = 3.839e33			# Units of [erg s^-1]		CGS
	R0 = 0.5 * 6.96e10		# Units of [cm]				CGS
	M0 = 0.7 * 1.99e33		# Units of [g]				CGS
	T0 = 1e5				# Units of [K]				SI, CGS
	P0 = 1e12				# Units of [Ba]				CGS
	rho0 = getRho(T0, P0)	# Units of [g cm^-3]		CGS

	n = int(2e6)	# Length of integration arrays

	rho = np.zeros(n)	# Initialising density array
	rho[0] = rho0

	epsilon = np.zeros(n)	# Initialising energy generation array
	epsilon[0] = L0 / M0

	# Initialising integration arrays
	m = np.zeros(n)
	m[0] = M0

	r = np.zeros(n)
	r[0] = R0

	P = np.zeros(n)
	P[0] = P0

	L = np.zeros(n)
	L[0] = L0

	T = np.zeros(n)
	T[0] = T0

	start = time.time()
	for i in range(n-1):
		# Progress counter
		percent = (i / float(n-1)) * 100
		sys.stdout.write('Progress: %4.2f %s\r' % (percent, '%'))
		sys.stdout.flush()

		if m[i] < 0 or r[i] < 0 or L[i] < 0 or getRho(T[i], P[i]) < 0 or P[i] < 0 or T[i] < 0:
			end = i
			print "Something went below zero. Stopping."
			print "Step:", i
			print "Mass:", m[i]/M0
			print "Radius:", r[i]/R0
			print "Luminosity:", L[i]/L0
			print "Density:", rho[i]/rho0
			print "Pressure:", P[i]/P0
			print "Temperature:", T[i]/T0
			break
		else:
			# Gradually increasing dm
			if i < 200000:
				dm = 1.39e23
			elif i < 400000:
				dm = 1.39e24
			elif i < 600000:
				dm = 1.39e25
			elif i < 400000:
				dm = 1.39e26
			else:
				dm = 1.39e27

			r[i+1] = r[i] - drdm(r[i],rho[i]) * dm
			P[i+1] = P[i] - dPdm(r[i],m[i]) * dm
			L[i+1] = L[i] - dLdm(T[i], rho[i]) * dm
			T[i+1] = T[i] - dTdm(T[i], L[i], r[i], opacity(T[i], rho[i])) * dm
			rho[i+1] = getRho(T[i],P[i])
			epsilon[i+1] = energyGeneration(T[i],rho[i])
			m[i+1] = m[i] - dm

	# Writing elapsed time upon completion
	finish = time.time()
	print "Time elapsed:", ((finish-start)/60.), " min"

	return r, P, L, T, m, rho, rho0, P0, T0, L0, M0, R0, epsilon, end

def plots():
	"""
	Function that plots the physical parameters.
	"""

	# Setting axis labels etc. in plots to the LaTeX font.
	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	params = {'text.usetex' : True,
			'font.size' : 22,
			'font.family' : 'lmodern',
			'text.latex.unicode': True,
			}
	plt.rcParams.update(params)

	# Calling the integration function
	r, P, L, T, m, rho, rho0, P0, T0, L0, M0, R0, epsilon, end = integration()

	
	# Plotting r(m)
	fig_r = plt.figure()
	ax_r = fig_r.add_subplot(111)

	ax_r.set_title('Position, $R_0 = 0.5R_{\odot}$')
	ax_r.set_xlabel('$m/M_0$')
	ax_r.set_ylabel('$r/R_0$')
	ax_r.plot(m[0:end]/M0,r[0:end]/R0)
	ax_r.grid('on')
	fig_r.tight_layout()
	
	# Plotting P(m)
	fig_P = plt.figure()
	ax_P = fig_P.add_subplot(111)

	ax_P.set_title('Pressure, $P_0 = 10^{16}$ Ba')
	ax_P.set_xlabel('$m/M_0$')
	ax_P.set_ylabel('$P/P_0$')
	ax_P.plot(m[0:end]/M0,P[0:end]/P0)
	ax_P.grid('on')
	fig_P.tight_layout()

	# Plotting L(m)
	fig_L = plt.figure()
	ax_L = fig_L.add_subplot(111)

	ax_L.set_title('Luminosity, $L_0 = L_{\odot}$')
	ax_L.set_xlabel('$m/M_0$')
	ax_L.set_ylabel('$L/L_0$')
	ax_L.plot(m[0:end]/M0,L[0:end]/L0)
	ax_L.grid('on')
	fig_L.tight_layout()

	# Plotting T(m)
	fig_T = plt.figure()
	ax_T = fig_T.add_subplot(111)

	ax_T.set_title('Temperature, $T_0 = 10^7$ K')
	ax_T.set_xlabel('$m/M_0$')
	ax_T.set_ylabel('$T/T_0$')
	ax_T.plot(m[0:end]/M0,T[0:end]/T0)
	ax_T.grid('on')
	fig_T.tight_layout()

	# Plotting rho(T, P)
	fig_p = plt.figure()
	ax_p = fig_p.add_subplot(111)

	ax_p.set_title('Density, $\\rho_0 =$ %.3f' % rho0)
	ax_p.set_xlabel('$m/M_0$')
	ax_p.set_ylabel('$\\rho/\\rho_0$')
	ax_p.plot(m[0:end]/M0,rho[0:end]/rho0)
	ax_p.grid('on')
	fig_p.tight_layout()

	# Plotting epsilon(T, rho)
	fig_e = plt.figure()
	ax_e = fig_e.add_subplot(111)

	ax_e.set_title('Energy generation per mass unit')
	ax_e.set_xlabel('$M/M_0$')
	ax_e.set_ylabel('$\\varepsilon$ [erg $g^{-1}$]')
	ax_e.plot(m[0:end]/M0,epsilon[0:end])
	ax_e.grid('on')
	fig_e.tight_layout()

	plt.show()

	pass

plots()
	
