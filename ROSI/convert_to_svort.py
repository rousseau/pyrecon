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
    args = input_parser.parse_args()

    print(type(args.results))
    res = job.load(open(args.results,"rb"))
    listSlice = res[0][1]
    nbSlice = len(listSlice)
    dir = args.output
    

    #load original data to get the data without normalisation
    listOriginal=[]
    for i in range(len(args.input_stacks)):
            print('----load images-----')
            im, inmask = loadStack(args.input_stacks[i], args.input_masks[i]) 
            print(args.input_masks[i])#,print('i',i))
            print(im.shape)
            new_mask = nib.Nifti1Image(inmask.get_fdata().squeeze(),inmask.affine)
            out = convert2Slices(im, new_mask, [], i,i)
            listOriginal+=out

    if not(os.path.exists(dir)):
        os.mkdir(dir)

    image,ori = same_order(listSlice,listOriginal)
    listSlice = np.concatenate(image)
    listOriginal = np.concatenate(ori)
    print(len(listSlice))
    print(len(listOriginal))
    index_original=[(s.get_indexSlice(),s.get_indexVolume()) for s in listOriginal]
    index_slice=[(s.get_indexSlice(),s.get_indexVolume()) for s in listSlice]
    print(index_original)
    print(index_slice)




    for i in range(0,nbSlice):
        islice = listSlice[i]
        index_slice = (islice.get_indexSlice(),islice.get_indexVolume())
        ior = index_original.index(index_slice)
        sliceor = listOriginal[ior]
        mask = islice.get_mask()
        affine = islice.get_slice().affine
        print('affine sign :',np.linalg.det(affine)<0)
        #islice.get_estimatedTransfo()
        sliceoriginal = sliceor.get_slice().get_fdata()
        #sliceor.get_slice().affine
        #sliceor.get_slice().get_fdata()
        #affine = islice.get_estimatedTransfo()
        sliceoriginal = listOriginal[ior].get_slice().get_fdata()
        dataslice = sliceoriginal
        #dataslice = sliceoriginal / np.quantile(sliceoriginal,0.99)
        nibslice = nib.Nifti1Image((dataslice),affine)
        nibslice.header.set_data_dtype(np.float32)
        nibmask = nib.Nifti1Image(mask,affine)
        nibmask.header.set_data_dtype(np.float32)
        nib.save(nibslice,dir+'/%d.nii.gz'%(i))
        #nib.save(nibmask,dir+'/mask_%d.nii.gz'%(i))

print(nibslice.header)
