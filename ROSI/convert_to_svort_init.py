##nb : nesvor ne gère pas les inténsité négative, donc si on veut utilisé ROSI avec Svort, il faut revoir la normalisation
##surtout que par defaut, toute les valeurs en dessous de zeros sont considérées comme ne faisant pas partie du masque et donc son enlevées....

from rosi.registration.tools import separate_slices_in_stacks
from rosi.registration.intersection import compute_cost_matrix
from rosi.registration.outliers_detection.feature import update_features
from rosi.registration.outliers_detection.outliers import sliceFeature
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
from rosi.registration.load import loadFromdir
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
        


    def add_input_slices(
        self,
        option_string="--input_slices",
        type=str,
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

    def add_output_mask(
        self,
        option_string="--output_mask",
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
    
    input_parser.add_input_slices(required=True) #load images
    #input_parser.add_input_mask(required=True) #load masks
    input_parser.add_results(required=True)
    input_parser.add_output(required=True) #load simulated transformation
    input_parser.add_output_mask(required=True)
    args = input_parser.parse_args()

    print(type(args.results))
    res = job.load(open(args.results,"rb"))
    listSlice = res[0][1]
    nbSlice = len(listSlice)
    dir = args.output
    dirmask = args.output_mask
    

    #load original data to get the data without normalisation
    listOriginal=[]


    if not(os.path.exists(dir)):
        os.makedirs(dir)
    if not(os.path.exists(dirmask)):
        os.makedirs(dirmask)

    #listOriginal = [file for file in os.listdir(args.input_slices) if not 'mask' in file]
    listOriginal = loadFromdir(args.input_slices)
    listErrorSlice = [sliceFeature(slicei.get_stackIndex(),slicei.get_indexSlice()) for slicei in listOriginal]
    
    squarre_error,number_point,intersection,union=compute_cost_matrix(listOriginal) 
    update_features(listOriginal,listErrorSlice,squarre_error,number_point,intersection,union)
    it=0
    while it < len(listErrorSlice):
            if listErrorSlice[it].get_mask_proportion()<0.1:
                    del listErrorSlice[it]
                    del listOriginal[it]
            else:
                    it+=1

    image,mask = separate_slices_in_stacks(listOriginal.copy())
    for m in image:
        listSliceNorm = listSliceNorm + m

    listOriginal = listSliceNorm
    print(len(listOriginal))
    print(len(listSlice))
                

    for i in range(0,nbSlice):
        islice = listSlice[i]
        index_slice = (islice.get_indexSlice(),islice.get_indexVolume())
        sliceor = listOriginal[i]
        #nib.load(os.path.join(args.input_slices,listOriginal[i]))
        mask = islice.get_mask()
        affine = islice.get_slice().affine
        #path_to_original = os.path.join(args.input_slices,listOriginal[i])
        sliceoriginal = islice.get_slice().get_fdata() * mask
        dataslice = sliceoriginal
        dataslice = sliceoriginal / np.quantile(sliceoriginal,0.99)
        nibslice = nib.Nifti1Image((sliceoriginal),affine)
        nibslice.header.set_data_dtype(np.float32)
        nibmask = nib.Nifti1Image(mask,affine)
        nibmask.header.set_data_dtype(np.float32)
        nib.save(nibslice,dir+'/%d.nii.gz'%(i))
        nib.save(nibmask,dirmask+'/mask_%d.nii.gz'%(i))

print(nibslice.header)
