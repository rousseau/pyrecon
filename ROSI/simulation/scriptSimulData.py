#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:25:56 2022

@author: mercier
"""

from simul3Ddata import extract_mask,simulateMvt
import nibabel as nib
from load import loadSlice
import numpy as np
import os
import argparse
import sys
from os import getcwd, path, mkdir
import six

#script to simulate LR image with motion from an HR image

#The function 'simulateMvt' simulates a LR image with inter-slice motion from an HR image.
#SimulateMVt take as parameters : the original HRImage, range motion for rotation, range motion for translation, upsampling parameters (to choose interslice resolution of LR image), image orientation, binary image corresponding to the mask and a boolean (Set to false if you don't want motion)
#And return : the LR image, mask of the LR image, parameters of transformation for each slices, rigid transformation for each slices.


class InputArgparser(object):

    def __init__(self,
                 description=None,
                 prog=None,
                 config_arg="--config"
                ):


        kwargs = {}

        self._parser = argparse.ArgumentParser(**kwargs)
        self._parser.add_argument(
            config_arg,
            help="Configuration file in JSON format.")
        self._parser.add_argument(
            "--version",
            action="version",           
        )
        self._config_arg = config_arg

    def get_parser(self):
        return self._parser

    def parse_args(self):

        # read config file if available
        if self._config_arg in sys.argv:
            self._parse_config_file()

        return self._parser.parse_args()

    def print_arguments(self, args, title="Configuration:"):
        
        for arg in sorted(vars(args)):
            
            vals = getattr(args, arg)

            if type(vals) is list:
                # print list element in new lines, unless only one entry in list
                # if len(vals) == 1:
                #     print(vals[0])
                # else:
                print("")
                for val in vals:
                    print("\t%s" % val)
            else:
                print(vals)
        


    def add_HR(
        self,
        option_string="--hr",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Mask(
        self,
        option_string="--mask",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Output(
        self,
        option_string="--output",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Motion(
        self,
        option_string="--motion",
        nargs="+",
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_Name(
        self,
        option_string="--name",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def _add_argument(self, allvars):

        # Skip variable 'self'
        allvars.pop('self')

        # Get name of argument to add
        option_string = allvars.pop('option_string')

        # Build dictionary for additional, optional parameters
        kwargs = {}
        for key, value in six.iteritems(allvars):
            kwargs[key] = value

        # Add information on default value in case provided
        if 'default' in kwargs.keys():

            if type(kwargs['default']) == list:
                txt = " ".join([str(i) for i in kwargs['default']])
            else:
                txt = str(kwargs['default'])
            txt_default = " [default: %s]" % txt

            # Case where 'required' key is given:
            if 'required' in kwargs.keys():

                # Only add information in case argument is not mandatory to
                # parse
                if kwargs['default'] is not None and not kwargs['required']:
                    kwargs['help'] += txt_default

            # Case where no such field was provided
            else:
                if kwargs['default'] is not None:
                    kwargs['help'] += txt_default

        # Add argument with its options
        self._parser.add_argument(option_string, **kwargs)



if __name__ == '__main__':
        
    root=getcwd()
    
    input_parser = InputArgparser()
    
    input_parser.add_HR(required=True) #load images
    input_parser.add_Mask(required=True) #load masks
    input_parser.add_Output(required=True) #load simulated transformation
    input_parser.add_Name(required=True)
    input_parser.add_Motion(required=True)
    args = input_parser.parse_args()


    HRnifti = nib.load(args.hr) #3D isotropic image
    Mask = nib.load(args.mask) #mask associated to the image
    binaryMask = extract_mask(Mask) #convert mask to a biniary mask
    #os.mkdir('/home/mercier/Documents/donnee/test/Grand5/')
    output = args.output
    name = args.name
    parameters_motion = args.motion
    motion=np.asarray(parameters_motion,dtype=np.float64)
    
    if not os.path.isdir(output):
        mkdir(output)
    
    
    LrAxNifti,AxMask,paramAx,transfoAx = simulateMvt(HRnifti,motion,motion,6,'axial',binaryMask.get_fdata(),True)#create an axial volume
    LrCorNifti,CorMask,paramCor,transfoCor = simulateMvt(HRnifti,motion,motion,6,'coronal',binaryMask.get_fdata(),True) #create a coronal volume
    LrSagNifti,SagMask,paramSag,transfoSag = simulateMvt(HRnifti,motion,motion,6,'sagittal',binaryMask.get_fdata(),True)#create a sagittal volume
    
    
    ##add noise to data
    sigma=np.random.uniform()*0.1
    print(sigma)
    
    mu=np.mean(LrAxNifti.get_fdata()[AxMask.get_fdata()>0])
    var=np.var(LrAxNifti.get_fdata()[AxMask.get_fdata()>0])
    print(mu*sigma)
    print(mu,var)
    data=LrAxNifti.get_fdata()+mu*np.random.normal(0,sigma,LrAxNifti.get_fdata().shape)
    LrAxNifti=nib.Nifti1Image(data, LrAxNifti.affine)
    
    nib.save(LrAxNifti,output + '/LrAxNifti_' +name +'.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask,output + '/LrAxNifti_' +name+ '_mask.nii.gz')
    np.save(output + '/paramAx_' +name+ '.npy',paramAx)
    np.save(output + '/transfoAx_' +name+ '.npy',transfoAx)
    
    mu=np.mean(LrCorNifti.get_fdata()[CorMask.get_fdata()>0])
    var=np.var(LrCorNifti.get_fdata()[CorMask.get_fdata()>0])
    print(mu*sigma)
    print(mu,var)
    data=LrCorNifti.get_fdata()+mu*np.random.normal(0,sigma,LrCorNifti.get_fdata().shape)
    LrCorNifti=nib.Nifti1Image(data, LrCorNifti.affine)
    
    nib.save(LrCorNifti, output +  '/LrCorNifti_' +name+ '.nii.gz')
    nib.save(CorMask,output +  '/LrCorNifti_' +name+ '_mask.nii.gz')
    np.save(output +  '/paramCor_' +name+ '.npy',paramCor)
    np.save(output +  '/transfoCor_' +name+ '.npy',transfoCor)
    
    mu=np.mean(LrSagNifti.get_fdata()[SagMask.get_fdata()>0])
    var=np.var(LrSagNifti.get_fdata()[SagMask.get_fdata()>0])
    print(mu*sigma)
    print(mu,var)
    data=LrSagNifti.get_fdata()+mu*np.random.normal(0,sigma,LrSagNifti.get_fdata().shape)
    LrSagNifti=nib.Nifti1Image(data, LrSagNifti.affine)
    
    nib.save(LrSagNifti,output +  '/LrSagNifti_' +name+ '.nii.gz')
    nib.save(SagMask,output +  '/LrSagNifti_' +name+ '_mask.nii.gz')
    np.save(output +  '/paramSag_' +name+ '.npy',paramSag)
    np.save(output +  '/transfoSag_' +name+ '.npy',transfoSag)

    LrAxNifti,AxMask,paramAx,transfoAx = simulateMvt(HRnifti,motion,motion,6,'axial',binaryMask.get_fdata(),False)#create an axial volume
    LrCorNifti,CorMask,paramCor,transfoCor = simulateMvt(HRnifti,motion,motion,6,'coronal',binaryMask.get_fdata(),False) #create a coronal volume
    LrSagNifti,SagMask,paramSag,transfoSag = simulateMvt(HRnifti,motion,motion,6,'sagittal',binaryMask.get_fdata(),False)#create a sagittal volume

    data=LrAxNifti.get_fdata()#+np.random.normal(0,sigma,LrAxNifti.get_fdata().shape)
    LrAxNifti=nib.Nifti1Image(data, LrAxNifti.affine)

    nib.save(LrAxNifti,output + '/LrAxNifti_nomvt.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask,output + '/AxMask_nomvt.nii.gz')
    np.save(output + '/paramAx_nomvt.npy',paramAx)
    np.save(output + '/transfoAx_nomvt.npy',transfoAx)
    
    data=LrCorNifti.get_fdata()#+np.random.normal(0,sigma,LrCorNifti.get_fdata().shape)
    LrCorNifti=nib.Nifti1Image(data, LrCorNifti.affine)
    
    nib.save(LrCorNifti, output +  '/LrCorNifti_nomvt.nii.gz')
    nib.save(CorMask,output +  '/CorMask_nomvt.nii.gz')
    np.save(output +  '/paramCor_nomvt.npy',paramCor)
    np.save(output +  '/transfoCor_nomvt.npy',transfoCor)
    
    data=LrSagNifti.get_fdata()#+np.random.normal(0,sigma,LrSagNifti.get_fdata().shape)
    LrSagNifti=nib.Nifti1Image(data, LrSagNifti.affine)
    
    nib.save(LrSagNifti,output +  '/LrSagNifti_nomvt.nii.gz')
    nib.save(SagMask,output +  '/SagMask_nomvt.nii.gz')
    np.save(output +  '/paramSag_nomvt.npy',paramSag)
    np.save(output +  '/transfoSag_nomvt.npy',transfoSag)


