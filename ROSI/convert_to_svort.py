##nb : nesvor ne gère pas les inténsité négative, donc si on veut utilisé ROSI avec Svort, il faut revoir la normalisation
##surtout que par defaut, toute les valeurs en dessous de zeros sont considérées comme ne faisant pas partie du masque et donc son enlevées....

import joblib as job
import nibabel as nib
from rosi.registration.sliceObject import SliceObject
import os
import numpy as np
from rosi.registration.load import loadStack, convert2Slices
import argparse
import six
import sys
from rosi.simulation.validation import same_order
from rosi.registration.outliers_detection.outliers import sliceFeature
from rosi.registration.outliers_detection.feature import update_features, detect_misregistered_slice
from rosi.registration.intersection import compute_cost_matrix, compute_cost_from_matrix
import pickle
from rosi.registration.outliers_detection.multi_start import correct_slice, removeBadSlice
#import sklearn.externals.joblib

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
        


    def add_input_stacks(
        self,
        option_string="--input_stacks",
        type=str,
        nargs="+",
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_input_mask(
        self,
        option_string="--input_masks",
        type=str,
        nargs="+",
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))
        
    def add_output(
        self,
        option_string="--output",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))


    def add_results(
        self,
        option_string="--results",
        type=str,
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_classifier(
        self,
        option_string="--classifier",
        type=str,
        default='my_model_nmse_inter_dice.pickle',
        help="",
        required=False,
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
    
    input_parser = InputArgparser()
    
    input_parser.add_input_stacks(required=True) #load images
    input_parser.add_input_mask(required=True) #load masks
    input_parser.add_results(required=True)
    input_parser.add_output(required=True) #load simulated transformation
    input_parser.add_classifier()
    args = input_parser.parse_args()


    print(type(args.results))
    res = job.load(open(args.results,"rb"))
    listSlice = res[0][1]
    nbSlice = len(listSlice)
    dir = args.output
    listOfOutliers = res[-1][1]
    print(listOfOutliers)
    load_model = pickle.load(open(args.classifier,'rb'))
    

    #load original data to get the data without normalisation
    listOriginal=[]
    i_image=0
    nb_remove=0
    i_prefix=0
    for i in range(len(args.input_stacks)):
        print(args.input_stacks[i])
        print('------------load images--------------------')
        print(args.input_stacks[i],args.input_masks[i])
        im, inmask = loadStack(args.input_stacks[i],args.input_masks[i]) 
        Affine = im.affine

        datamask = inmask.get_fdata().squeeze()
        ##check mask and image size : 
        if datamask.shape==im.get_fdata().shape:
            ##check mask and image size : 
            if datamask.shape==im.get_fdata().shape:
                mask = nib.Nifti1Image(datamask, inmask.affine)
            
                if  i==0:
                    nx_img1=Affine[0:3,0].copy()
                    ny_img1=Affine[0:3,1].copy()
                    nz_img1=np.cross(nx_img1,ny_img1)
                    nx_img1_norm=nx_img1/np.linalg.norm(nx_img1)
                    ny_img1_norm=ny_img1/np.linalg.norm(ny_img1)
                    nz_img1_norm=nz_img1/np.linalg.norm(nz_img1)
                    output = convert2Slices(im,mask,[],i_image,i_image)
                    listOriginal+=output
                    i_image=i_image+1
            
                else:
                    nx=Affine[0:3,0].copy()
                    ny=Affine[0:3,1].copy()
                    nz=np.cross(nx,ny)
                    nz_norm=nz/np.linalg.norm(nz)
                    
                    orz=np.abs(np.dot(nz_norm,nz_img1_norm))
                    ory=np.abs(np.dot(nz_norm,ny_img1_norm))
                    orx=np.abs(np.dot(nz_norm,nx_img1_norm))
                
                    if max(orx,ory,orz)==orx:
                        output = convert2Slices(im,mask,[],1,i_image)
                        listOriginal+=output
                        print('orx :', orx, 'ory :', ory, 'orz :', orz)
                        print(i, ' : Coronal')
                        i_image=i_image+1
                
                    elif max(orx,ory,orz)==ory:
                        output = convert2Slices(im,mask,[],2,i_image)
                        listOriginal+=output
                        print('orx :', orx, 'ory :', ory, 'orz :', orz)
                        print(i ,' : Sagittal')
                        i_image=i_image+1
                
                    else:
                        output = convert2Slices(im,mask,[],0,i_image)
                        listOriginal+=output
                        print('orx :', orx, 'ory :', ory, 'orz :', orz)
                        print(i , ' : Axial')
                        i_image=i_image+1
                
                print('i_image',i_image)
                
                
            else :
                i_prefix = i - nb_remove
                del list_prefixImage[i_prefix]
                print(list_prefixImage)
                nb_remove=nb_remove+1

    listFeatures = [sliceFeature(s.get_stackIndex(),s.get_indexSlice()) for s in listSlice]
    squarre_error,nbpoint_matrix,intersection_matrix,union_matrix=compute_cost_matrix(listSlice)
    matrix = np.array([squarre_error,nbpoint_matrix,intersection_matrix,union_matrix])
    update_features(listSlice,listFeatures,squarre_error,nbpoint_matrix,intersection_matrix,union_matrix)
    set_r = detect_misregistered_slice(listSlice,matrix,load_model,0.5)
    listOfOutliers = removeBadSlice(listSlice,set_r)

    listErrorSlice = [sliceFeature(s.get_stackIndex(),s.get_indexSlice()) for s in listOriginal]
    squarre_error,number_point,intersection,union=compute_cost_matrix(listOriginal) 
    update_features(listOriginal,listErrorSlice,squarre_error,number_point,intersection,union)
    it=0
    while it < len(listErrorSlice):
            if listErrorSlice[it].get_mask_proportion()<0.1:
                    del listErrorSlice[it]
                    del listOriginal[it]
            else:
                    it+=1
    
    if not(os.path.exists(dir)):
        os.makedirs(dir)

    #image,ori = same_order(listSlice,listOriginal)
    #listSlice = np.concatenate(image)
    #listOriginal = np.concatenate(ori)
    print(len(listSlice))
    print(len(listOriginal))
    index_original=[(s.get_indexVolume(),s.get_indexSlice()) for s in listOriginal]
    index_slice=[(s.get_indexVolume(),s.get_indexSlice()) for s in listSlice]
    print(index_original)
    print(index_slice)


    for i in range(0,nbSlice):
        islice = listSlice[i]
        index_slice = (islice.get_indexVolume(),islice.get_indexSlice())
        ior=i
        #ior = index_original.index(index_slice)
        sliceor = listOriginal[ior]
        mask = islice.get_mask()
        affine = islice.get_slice().affine
        print('affine sign :',np.linalg.det(affine)<0)
        sliceoriginal = sliceor.get_slice().get_fdata()
        if np.linalg.det(affine)<0 : #if the determinant is negatif, performs horizontal flip on the image
            #w,h,d = data.shape
            #affine[:,0] *=-1
            #affine[0,3] -=((w-1)/2)
            #new_affine = affine
            sliceoriginal = np.flip(sliceor.get_slice().get_fdata(),0)
        #islice.get_estimatedTransfo()
        
        #sliceoriginal=islice.get_fdata()
        #sliceor.get_slice().affine
        #sliceor.get_slice().get_fdata()
        affine = islice.get_estimatedTransfo()
        #sliceoriginal = sliceor.get_slice().get_fdata() 
        #dataslice = sliceoriginal
        dataslice = (sliceoriginal / np.quantile(sliceoriginal,0.99))#*mask
        nibslice = nib.Nifti1Image((dataslice),affine)
        nibslice.header.set_data_dtype(np.float32)
        nibmask = nib.Nifti1Image(mask,affine)
        nibmask.header.set_data_dtype(np.float32)
        #if not index_slice in listOfOutliers : 
        nib.save(nibslice,dir+'/%d.nii.gz'%(i))
        nib.save(nibmask,dir+'/mask_%d.nii.gz'%(i))
        #else : 
        #    print("this is an outlier slice")

print(nibslice.header)
