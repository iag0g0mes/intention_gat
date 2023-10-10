import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Tuple, NoReturn

from scipy.signal import savgol_filter

class SavitzkyGolay(object):

	def __init__(self, window_length:int, poly:int):

		self.size=window_length
		self.poly=poly

	def set_window_size(self, size):
		self.size=size

		if self.size%2==0:
			self.size = self.size+1

	def process(self, traj:np.ndarray)->np.ndarray:
		
		x = traj[:,0]
		y = traj[:,1]
	
		_x = savgol_filter(x=x, 
							window_length=self.size, 
							polyorder=self.poly, 
							deriv=0,
							delta=0.1)
		_y = savgol_filter(x=y, 
							window_length=self.size, 
							polyorder=self.poly, 
							deriv=0,
							delta=0.1)
		

		return np.squeeze(np.dstack([_x, _y]))

	def filter(self, vector:np.ndarray)->np.ndarray:
		vector = np.squeeze(vector)

		x = vector[:,0]
		y = vector[:,1]

		p = np.dstack((np.zeros(len(x)), x, y))
		p = np.squeeze(p)

		return self.process(p)

	def filter2(self, vector:np.ndarray)->np.ndarray:
		vector = np.squeeze(vector)

		result = [savgol_filter(x=vector[:, i], 
						  window_length=self.size, 
						  polyorder=self.poly)\
				 for i in range(0, vector.shape[1])]

		return np.squeeze(np.dstack(result))

class EKF(object):

	def __init__(self):

		self.init()

	def init(self):
		#state vector
		self.X = np.array([[0],    # x
						   [0],    # y
						   [0],    # v_x
						   [0],    # v_y
						   [0],    # a_x
						   [0],    # a_y
						   [0],    # j_x
						   [0]])   # j_y
		#identity matrix
		self.I = np.eye(8)
		#process covariance  
		self.P = 1000*self.I
		#jacobian h(x) 
		self.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
						   [0., 1., 0., 0., 0., 0., 0., 0.]])
		#covariance noise process
		coeficient_Q = np.array([0.1, # x
								 0.1, # y
								 0.5, # v_x
								 0.5, # v_y
								 0.1, # a_x
								 0.1, # a_y
								 0.5, # j_x
								 0.5])# j_y
								 
		self.Q = np.eye(8)*coeficient_Q
		#covariance noise observation
		self.R = np.array([[1., 0.],  # x_obs
						   [0., 1.]]) # y_obs

	def _update(self, z:np.array) -> np.ndarray:

		z = z.reshape(2,1) # x_obs, y_obs

		# estimate innovation
		y = z - np.dot(self.H, self.X)
		# estimate innovation covariance 
		S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
		# estimate the near-optiman kalman gain
		K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.inv(S))
		# update state
		self.X = self.X + np.dot(K, y)
		# update process covariance
		self.P = np.dot((self.I - np.dot(K, self.H)), self.P)


		return self.X.reshape(8)

	def _predict(self, dt:float):

		"""
			A) prediction step
			  - Equations:

			x = xo + vx*dt
			y = yo + vy*dt
			v_x = vo_x + ax*t
			v_y = vo_y + ay*t
			a_x = (v_x - vo_x)/dt
			a_y = (v_y - vo_y)/dt
			j_x = (a_x - ao_x)/dt
			j_y = (a_y - ao_y)/dt
		"""
		# # 		# x   y  v_x v_y a_x  a_y j_x j_y
		# jac_X = [[1., 0., dt, 0., 0., 0., 0., 0.], # x
		# 		 [0., 1., 0., dt, 0., 0., 0., 0.], # y
		# 		 [0., 0., 1., 0., dt, 0., 0., 0.], # v_x
		# 		 [0., 0., 0., 1., 0., dt, 0., 0.], # v_y
		# 		 [0., 0., 1./dt, 0., 0., 0., 0., 0.], # a_x
		# 		 [0., 0., 0., 1./dt, 0., 0., 0., 0.], # a_y
		# 		 [0., 0., 0., 0., 1./dt, 0., 0., 0.], # j_x
		# 		 [0., 0., 0., 0., 0., 1./dt, 0., 0.]] # j_y

				# x   y  v_x v_y a_x  a_y j_x j_y
		jac_X = [[1., 0., dt, 0., 0., 0., 0., 0.], # x
				 [0., 1., 0., dt, 0., 0., 0., 0.], # y
				 [1./dt, 0., 0., 0., 0., 0., 0., 0.], # v_x
				 [0., 1./dt, 0., 0., 0., 0., 0., 0.], # v_y
				 [0., 0., 1./dt, 0., 0., 0., 0., 0.], # a_x
				 [0., 0., 0., 1./dt, 0., 0., 0., 0.], # a_y
				 [0., 0., 0., 0., 1./dt, 0., 0., 0.], # j_x
				 [0., 0., 0., 0., 0., 1./dt, 0., 0.]] # j_y

		# estimate P (process covariance) (without control input U)
		self.P = np.dot(np.dot(jac_X, self.P), np.transpose(jac_X)) 

		# estimate new state vector X (prediction)
		_x = self.X[0] + self.X[2]*dt
		_y = self.X[1] + self.X[3]*dt
		_v_x = (_x - self.X[0])/dt#self.X[2] + self.X[4]*dt
		_v_y = (_y - self.X[1])/dt#self.X[3] + self.X[5]*dt
		_a_x = (_v_x - self.X[2])/dt
		_a_y = (_v_y - self.X[3])/dt
		_j_x = (_a_x - self.X[4])/dt
		_j_y = (_a_y - self.X[5])/dt

		self.X = np.array([_x, 
						   _y, 
						   _v_x,
						   _v_y,
						   _a_x,
						   _a_y,
						   _j_x,
						   _j_y]).reshape((8,1))
		
	# def _predict2(self, dt:float):

	# 	"""
	# 		A) prediction step
	# 		  - Equations:

	# 		x = xo + vx*dt + (axt^2)/2
	# 		y = yo + vy*dt + (ayt^2)/2
	# 		v_x = vo_x + ax*t
	# 		v_y = vo_y + ay*t
	# 		a_x = (v_x - vo_x)/dt
	# 		a_y = (v_y - vo_y)/dt
	# 		j_x = (a_x - ao_x)/dt
	# 		j_y = (a_y - ao_y)/dt
	# 	"""
	# 			# x   y  v_x    v_y      a_x        a_y     j_x  j_y
	# 	jac_X = [[1., 0., dt,    0., (dt*dt)/2.,     0.    , 0., 0.], # x
	# 			 [0., 1., 0.,    dt,     0.,     (dt*dt)/2., 0., 0.], # y
	# 			 [0., 0., 1.,    0.,     dt,         0.    , 0., 0.], # v_x
	# 			 [0., 0., 0.,    1.,     0.,         dt,     0., 0.], # v_y
	# 			 [0., 0., 1./dt, 0.,     0.,         0.,     0., 0.], # a_x
	# 			 [0., 0., 0.,  1./dt,    0.,         0.,     0., 0.], # a_y
	# 			 [0., 0., 0.,    0.,   1./dt,        0.,     0., 0.], # j_x
	# 			 [0., 0., 0.,    0.,     0.,       1./dt,    0., 0.]] # j_y

	# 	# estimate P (process covariance) (without control input U)
	# 	self.P = np.dot(np.dot(jac_X, self.P), np.transpose(jac_X)) 

	# 	# estimate new state vector X (prediction)
	# 	_x = self.X[0,0] + self.X[2,0]*dt + (self.X[4,0]*dt*dt)/2.
	# 	_y = self.X[1,0] + self.X[3,0]*dt + (self.X[5,0]*dt*dt)/2.
	# 	_v_x = self.X[2,0] + self.X[4,0]*dt
	# 	_v_y = self.X[3,0] + self.X[5,0]*dt
	# 	_a_x = (_v_x - self.X[2,0])/dt
	# 	_a_y = (_v_y - self.X[3,0])/dt
	# 	_j_x = (_a_x - self.X[4,0])/dt
	# 	_j_y = (_a_y - self.X[5,0])/dt

	# 	self.X = np.array([_x, 
	# 					   _y, 
	# 					   _v_x,
	# 					   _v_y,
	# 					   _a_x,
	# 					   _a_y,
	# 					   _j_x,
	# 					   _j_y]).reshape((8,1))

	def clean(self)->NoReturn:
		self.init()

	def process(self, traj:np.ndarray) -> np.ndarray:
		
		self.X[0] = traj[0, 1] #x
		self.X[1] = traj[0, 2] #y
		self.X[2] = (traj[1, 1] - traj[0,1])/(traj[1, 0] - traj[0,0]) #vx
		self.X[3] = (traj[1, 2] - traj[0,2])/(traj[1, 0] - traj[0,0]) #vy
		self.X[4] = np.random.randint(-2,2,1)
		self.X[5] = np.random.randint(-2,2,1)

		last_t = traj[0,0]

		result = []

		for tj in traj[1:,:]:
			dt = tj[0]  - last_t
			self._predict(dt = dt)
			x = self._update(z=tj[1:])
			result.append(x)
			last_t = tj[0]

		return np.asarray([result])


class PlainFeatures(object):

	def __init__(self):
		pass

	def get_velocity(self, traj:np.ndarray)->np.ndarray:
		
		x = traj[:, 1]
		y = traj[:, 2]
		t = traj[:, 0]

		vel_x = np.ediff1d(x)/np.ediff1d(t)
		vel_y = np.ediff1d(y)/np.ediff1d(t)
		

		vel = np.sqrt(np.power(vel_x,2) + np.power(vel_y,2))

		return vel_x, vel_y, vel

	def get_acceleration(self, traj:np.ndarray)->np.ndarray:
		
		vel_x, vel_y, _ = self.get_velocity(traj)

		#0,1 -> v0
		#1,2 -> v1, a0
		t = traj[1:,0]

		acc_x = np.ediff1d(vel_x)/np.ediff1d(t)
		acc_y = np.ediff1d(vel_y)/np.ediff1d(t)
		

		acc = np.sqrt(np.power(acc_x,2) + np.power(acc_y,2))

		return acc_x, acc_y, acc


	def get_jerk(self, traj:np.ndarray)->np.ndarray:
		
		acc_x, acc_y, _ = self.get_acceleration(traj)

		#0,1 -> v0
		#1,2 -> v1, a0
		#2,3 -> v2, a1, j0
		t = traj[2:,0]

		j_x= np.ediff1d(acc_x)/np.ediff1d(t)
		j_y= np.ediff1d(acc_y)/np.ediff1d(t)

		j = np.sqrt(np.power(j_x,2) + np.power(j_y,2))

		return j_x, j_y, j


	def process(self, traj:np.ndarray) -> np.ndarray:
		
		v_x, v_y, v = self.get_velocity(traj=traj)
		a_x, a_y, a = self.get_acceleration(traj=traj)
		j_x, j_y, j = self.get_jerk(traj=traj)
		x = traj[:,1]
		y = traj[:,2]
		
		result = np.dstack((x[3:],
							y[3:],
							v_x[2:],
							v_y[2:],
							a_x[1:],
							a_y[1:],
							j_x,
							j_y))

		return np.asarray([result])