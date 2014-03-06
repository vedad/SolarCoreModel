import numpy as np
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

	# Finds the index of the minimum difference values, so the most relevant khappa can be used.
	i = np.argmin(diffT)
	j = np.argmin(diffR)
	
	khappa = 10**(logK[i,j])
	return khappa


def energyGeneration(T, rho):
	"""
	Function to find the full energy generation per
	unit mass from the three PP chain reactions.
	"""
#	mu = 1.6605e-27		# Units of [kg]		SI
	mu = 1.6605e-24		# Units of [g]		CGS
#	mu = 1				# Units of [u]

	# Abundancy of different elements
	X = 0.7				# Hydrogen (ionised)
	Y = 0.29			# Helium 4 (ionised)
	Y_3 = 1e-10			# Helium 3 (ionised)
	Z = 0.01			# Heavier elements than the above
	Z_7Li = 1e-5		# Lithium 7 (part of Z)
	Z_7Be = 1e-5		# Beryllium 7 (part of Z)
	
	### Energy values (Q) ###
	# Energy values of nuclear reactions in PP I [MeV]
	Q_pp = 1.177 + 5.949; Q_3He3He = 12.86

	# Energy values of nuclear reactions in PP II [MeV]
	Q_3He4He = 1.586; Q_e7Be = 0.049; Q_p7Li = 17.346

	# Energy values of nuclear reactions in PP III [MeV]
	Q_p7Be= 0.137 + 8.367 + 2.995

	### Number densities (n) ###
	n_p = X * rho / mu					# Hydrogen
	n_3He = Y_3 * rho / (3 * mu)		# Helium 3
	n_4He = Y * rho / (4 * mu)			# Helium 4
	n_7Be = Z_7Be * rho / (7 * mu)		# Beryllium 7
	n_e = n_p + 2 * n_4He				# Electron
	n_7Li = Z_7Li * rho / (7 * mu)		# Lithium 7

	### Reaction rates (lambda) ### Units of reactions per second per cubic cm. [cm^3 s^-1]
	T9 = T / (1e9)

	l_pp = 4.01e-15 * T9**(-2./3) * np.exp(-3.380 * T9**(-1./3)) * (1 + 0.123 * T9**(1./3)
			+ 1.09 * T9**(2./3) + 0.938 * T9)

	l_3He3He = 6.04e10 * T9**(-2./3) * np.exp(-12.276 * T9**(-1./3)) * (1 + 0.034 * T9**(1./3) -
			0.522 * T9**(2./3) - 0.124 * T9 + 0.353 * T9**(4./3) + 0.213 * T9**(5./3))

	a = 1 + 0.0495 * T9 # Defined for efficiency in l_3He4He
	l_3He4He = 5.61e6 * a**(-5./6) * T9**(-2./3) * np.exp(-12.826 * a**(1./3) * T9**(-1./3))

	l_e7Be = 1.34e-10 * T9**(-1./2) * (1 - 0.537 * T9**(1./3) + 3.86 * T9**(2./3)\
			+ 0.0027 * T9**(-1.) * np.exp(2.515e-3 / T9))

	a = 1 + 0.759 * T9 # Defined for efficiency in l_p7Li
	l_p7Li = 1.096e9 * T9**(-2./3) * np.exp(-8.472 * T9**(-1./3)) - 4.830e8 * a**(-5./6)\
	* T9**(-2./3) * np.exp(-8.472 * a**(1./3) * T9**(-1./3))

	l_p7Be = 3.11e5 * T9**(-2./3) * np.exp(-10.262 * T9**(-1./3))

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
	
	epsilon = 1.602e-6 / 6.022e23
	print epsilon
	return epsilon # Conversion from MeV to ergs for CGS

def rho(T, P):
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
#	sigma = 5.67e-8		# Units of [W m^-2 K^-4]		SI
	sigma = 5.67e-5	# Units of [erg cm^-2 s^-1 K^-4]	CGS

	# Boltzmann constant
#	k = 1.38e-23			# Units of [J K^-1]			SI
	k = 1.38e-16			# Units of [erg K^-1]		CGS

	# Atonmic mass unit
#	m_u = 1.6605e-27		# Units of [kg]				SI
	m_u = 1.6605e-24		# Units of [g]				CGS
#	m_u = 1					# Units of [u]

	# Speed of light
#	c = 3e8					# Units of [m s^-1]			SI
	c = 3e10				# Units of [cm s^-1]		CGS

	# Radiation constant
	a = 4 * sigma / c

	# Average molecular weight
	mu = 1./(2 * X + 3. / 4 * Y + Z / 2)

	rho = mu * m_u / (k * T) * (P - a/3. * T*T*T*T)
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
#	G = 6.67e-11	# Units of [N m kg^-2]			SI
	G = 6.67e-8		# Units of [cm^3 g^-1 s^-2]		CGS
	return - G * m / (4 * np.pi * r * r * r * r)

def dLdm(T, rho):
	"""
	Calculates the right-hand side of dL/dm.
	"""
	return energyGeneration(T, rho)

def dTdm(T, L, r, khappa):
	"""
	Calculates the right-hand side of dT/dm.
	"""
#	sigma = 5.67e-8			# Units of [W m^-2 K^-4]			SI
	sigma = 5.67e-5			# Units of [erg cm^-2 s^-1 K^-4]	CGS

	return -3 * khappa * L / (256 * np.pi*np.pi * sigma * r*r*r*r * T*T*T)


def integration(dm):
	"""
	Function that integrates the equation governing
	the internal structure of the radiative core
	of the Sun.
	"""
	
#	L0 = 3.839e26			# Units of [J s^-1]			SI
	L0 = 3.839e33			# Units of [erg s^-1]		CGS

#	R0 = 0.5 * 6.96e8		# Units of [m]				SI
	R0 = 0.5 * 6.96e10		# Units of [cm]				CGS

#	M0 = 0.7 * 1.99e30		# Units of [kg]				SI
	M0 = 0.7 * 1.99e33		# Units of [g]				CGS

	T0 = 1e5				# Units of [K]				SI, CGS

#	P0 = 1e11				# Units of [Pa]				SI
	P0 = 1e12				# Units of [Ba]				CGS

	rho0 = rho(T0, P0)		# Units of [g cm^-3]		CGS

	n = int(round(M0 / dm))	# Length of integration arrays

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


	for i in range(n-1):
		print T[i]
		if r[i] < 0 or L[i] < 0 or m[i] < 0:
			break
		else:
			r[i+1] = r[i] - drdm(r[i], rho(T[i], P[i])) * dm
			P[i+1] = P[i] - dPdm(r[i],m[i]) * dm
			L[i+1] = L[i] - dLdm(T[i], rho(T[i], P[i])) * dm
			T[i+1] = T[i] - dTdm(T[i], L[i], r[i], opacity(T[i], rho(T[i], P[i]))) * dm
			m[i+1] = m[i] - dm

	return r, P, L, T, m


#integration(1.39e29)


def plots():
	
	r, P, L, T, m = integration(1.39e30)
	
	# Plotting r(m)
	fig_r = plt.figure()
	ax_r = fig_r.add_subplot(111)

	ax_r.set_title('$r(m)$')
	ax_r.set_xlabel('$m$')
	ax_r.set_ylabel('$r$')
	ax_r.plot(m,r)
	
	# Plotting P(m)
	fig_P = plt.figure()
	ax_P = fig_P.add_subplot(111)

	ax_P.set_title('$P(m)$')
	ax_P.set_xlabel('$m$')
	ax_P.set_ylabel('$P$')
	ax_P.plot(m,P)

	# Plotting L(m)
	fig_L = plt.figure()
	ax_L = fig_L.add_subplot(111)

	ax_L.set_title('$L(m)$')
	ax_L.set_xlabel('$m$')
	ax_L.set_ylabel('$L$')
	ax_L.plot(m,L)

	# Plotting T(m)
	fig_T = plt.figure()
	ax_T = fig_T.add_subplot(111)

	ax_T.set_title('$T(m)$')
	ax_T.set_xlabel('$m$')
	ax_T.set_ylabel('$T$')
	ax_T.plot(m,T)

	plt.show()

	pass

plots()
	
