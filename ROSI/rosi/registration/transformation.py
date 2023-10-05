import numpy as np
from numpy.linalg import inv,eig
from scipy.linalg import expm

def rotationCentre(mask : np.array) -> np.array:
    """
    Compute the barycentre 
    """    

    index = np.where(mask>0)
    barycenter = np.sum(index,axis=1)/(np.sum(mask)) #2D barycentre
    barycenter = np.concatenate((barycenter[0:2],np.array([0,1]))) #barycenter in homgenous coordinate
    return barycenter


def rigidMatrix(parameters : np.array) -> np.array:
      
    """"
    Calculate the rigidMatrix with 6 parameters. 
    
    parameters = [theta_1,theta_2,theta_3,t_1,t_2,t_3].
    
    Rotation: 
    theta_1,theta_2,theta_3 are in degrees, eulerian representation
    Considering a reference system in 3D defined by three axes, x,y,z: 
    theta_1 defines a rotation around x
    theta_2 defines a rotation about y
    theta_3 defines a rotation about z
    Help to understand the rotation matrix : https://en.wikipedia.org/wiki/Rotation_matrix 
    
    translation : 
    t_1,t_2,t_3 in mm
    """
   
    #convert angles into radiant
    theta_1 = np.pi*(parameters[0]/180.0)
    theta_2 = np.pi*(parameters[1]/180.0)
    theta_3 = np.pi*(parameters[2]/180.0)

    
    rigide = np.eye(4)

    #translation
    rigide[0:3,3] = parameters[3:6]
    
    cosg=np.cos(theta_1)
    cosa=np.cos(theta_3)
    cosb=np.cos(theta_2)
    sing=np.sin(theta_1)
    sina=np.sin(theta_3)
    sinb=np.sin(theta_2)
    
    
    #rotation matrix, rotation around the axe x, y and k
    rigide[0,0] = cosa*cosb
    rigide[1,0] = cosa*sinb*sing-sina*cosg
    rigide[2,0] = cosa*sinb*cosg+sina*sing
    rigide[0,1] = sina*cosb
    rigide[1,1] = sina*sinb*sing+cosa*cosg
    rigide[2,1] = sina*sinb*cosg-cosa*sing
    rigide[0,2] = -sinb
    rigide[1,2] = cosb*sing
    rigide[2,2] = cosb*cosg
    
    return rigide


def ParametersFromRigidMatrix(rigidMatrix : np.array) -> np.array:
   
    """
    Find parameters of a rigidMatrix (3 parameters for rotation and 3 parameters for transaltion)
    """
    
    p=np.zeros(6)
    
    #translation (in mm)
    p[3]=rigidMatrix[0,3]
    p[4]=rigidMatrix[1,3]
    p[5]=rigidMatrix[2,3]
    
    #rotation (in degree)
    theta_2=np.arcsin(-rigidMatrix[0,2])
    theta_1=np.arctan2(rigidMatrix[1,2]/np.cos(theta_2),rigidMatrix[2,2]/np.cos(theta_2))
    theta_3=np.arctan2(rigidMatrix[0,1]/np.cos(theta_2),rigidMatrix[0,0]/np.cos(theta_2))
    p[0]=(180.0*theta_1)/np.pi
    p[1]=(180.0*theta_2)/np.pi
    p[2]=(180.0*theta_3)/np.pi
 
    
    return p

def debug_meanMatrix(mean_rotation : np.array,
                     mean_translation : np.array,
                     parameters : np.array) -> bool: 
    """
    Debug meanMatrix function
    """
   
    np.real(mean_rotation)
    mean_rigid=np.zeros((4,4))
    mean_rigid[3,3]=1
    mean_rigid[0:3,0:3]=mean_rotation
    mean_rigid[0:3,3]=mean_translation
    
    check_matrix=rigidMatrix(parameters)
    
    #check that the rotation are the same
    print('res',mean_rigid)
    print('r_mean',mean_rotation)
    
    #check if matrix are identiqual
    equal=np.all(mean_rigid==check_matrix)
    print(equal)

    return equal

def log_cplx(x):
    res = np.log(abs(x)) + np.angle(x)*1j
    return  res



def IsItRigid(matrix : np.array) -> bool : 
    """
    Take a 4x4 homogenous matrix and check : 
    1. That the matrix is indeed homogenous
    2. The the rotation part is indeed a rotation 
       2.1 Check if the matrix is orthogonal
       2.2 Check the matrix determinant
    """

    rotation = matrix[0:3,0:3]
    homogeneous = matrix[3,:]

    check_homogenity = np.all(homogeneous == np.array([0,0,0,1]))
    check_orthogonality = np.all((rotation@rotation.T - np.eye(3)) < 1e-6)
    check_det = np.linalg.det(rotation) > 0
 
    rigidity = check_homogenity and check_orthogonality and check_det

    return rigidity
