import numpy as np

def rotationCenter(mask):
    """
    
    Compute the barycentre
    
    Inputs :
    mask : 2D image
        binary image which indicates the position of the brain

    Outputs : 
    centerw : 2xD vector
   
    """    
    index = np.where(mask>0)
    center = np.sum(index,axis=1)/(np.sum(mask))
    centerw = np.concatenate((center[0:2],np.array([0,1])))
    #centerw = sliceaffine @ centerw 
       
    return centerw


def rigidMatrix(parameters):
    """
    Compute the rigidMatrix with 6 parameters. The first three parameters correspond to the rotation and the last three to the translation.

    Inputs : 
    parameters : The parameters of the rigid transformation, the first three parameters correspond to the rotation and the last three to the translation

    Outputs : 
    rigide : 4x4 matrix
    The translation matrix in homogenous coordinates

    """
   
    #convert angles into radiant
    gamma = np.pi*(parameters[0]/180.0)
    beta = np.pi*(parameters[1]/180.0)
    alpha = np.pi*(parameters[2]/180.0)

    
    rigide = np.eye(4)
    rigide[0:3,3] = parameters[3:6]
    
    cosg=np.cos(gamma)
    cosa=np.cos(alpha)
    cosb=np.cos(beta)
    sing=np.sin(gamma)
    sina=np.sin(alpha)
    sinb=np.sin(beta)
    
    
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


def ParametersFromRigidMatrix(rigidMatrix):
   
    """
    Find parameters associated with a rigidMatrix (3 parameters for rotation and 3 parameters for transaltion)
    """
    
    p=np.zeros(6)
    
    p[3]=rigidMatrix[0,3]
    p[4]=rigidMatrix[1,3]
    p[5]=rigidMatrix[2,3]
    
    beta=np.arcsin(-rigidMatrix[0,2])
    gamma=np.arctan2(rigidMatrix[1,2]/np.cos(beta),rigidMatrix[2,2]/np.cos(beta))
    alpha=np.arctan2(rigidMatrix[0,1]/np.cos(beta),rigidMatrix[0,0]/np.cos(beta))
    p[0]=(180.0*gamma)/np.pi
    p[1]=(180.0*beta)/np.pi
    p[2]=(180.0*alpha)/np.pi
 
    
    return p