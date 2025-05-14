#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:52:47 2022

@author: mercier
"""

import sys
import six
import argparse
import numpy as np

#from https://github.com/gift-surg/NiftyMIC/blob/553bce0824e7b40cd221897b683142d9aeee77d8/niftymic/utilities/input_arparser.py


class InputArgparser(object):
    
    def __init__(self,description=None,config_arg="--config"):
        kwargs = {}
        self._parser = argparse.ArgumentParser(**kwargs)
        self._config_arg = config_arg

    def parse_args(self):

        # read config file if available
        if self._config_arg in sys.argv:
            self._parse_config_file()

        return self._parser.parse_args()
    
    def error(self,error_message):

        return self._parser.error(error_message)
        
    def add_input_stacks(
        self,
        option_string="--input-stacks",
        nargs="+",
        help="image data in nii.gz",
        default=None,
        required=True,
                    ):
        self._add_argument(dict(locals()))
        
    def add_input_masks(
        self,
        option_string="--input-masks",
        nargs="+",
        help="mask of the data in nii.gz",
        default=None,
        required=True,
                    ):
        self._add_argument(dict(locals()))   
        
    def add_nomvt(
        self,
        option_string="--nomvt",
        nargs="+",
        help="image data in nii.gz",
        default=None,
        required=False,
                        ):
        self._add_argument(dict(locals()))
            
    def add_nomvt_mask(
        self,
        option_string="--nomvt-mask",
        nargs="+",
        help="mask of the data in nii.gz",
        default=None,
        required=False,
        ):
        self._add_argument(dict(locals()))  
        

    def add_output(
        self,
        option_string="--output",
        type=str,
        help="joblib wich contains all the output data, there is no need to provide extension, ex 'result' ",
        default=None,
        required=True,
                   ):
        self._add_argument(dict(locals()))

            
    def add_transformation(
        self,
        option_string="--transformation",
        nargs="+",
        help="theorical transformation for the simulated data",
        default=None,
        required=False,
                    ):
        self._add_argument(dict(locals()))

    def add_tre(
        self,
        option_string="--tre",
        type=bool,
        default=0,
        help="Set to 1 if you are using the simulated data generated from (script_of_simulation) and want to know the TRE (Target Registration Error) obtained on each slice ",
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
        
    def add_isimplex(
        self,
        option_string="--initial-simplex",
        type=float,
        default=4,
        help="If optimisation is Nelder-Mead: size of the initial simplex used for optimisation, default=4mm",
        required=False,
        ):
        self._add_argument(dict(locals()))
    
    def add_fsimplex(
        self,
        option_string="--final-simplex",
        type=float,
        default=2,
        help="If optimisation is Nelder-Mead: size of the final simplex used for optimisation, default=0.25mm",
        required=False,
        ):
        self._add_argument(dict(locals()))
    
    def add_localConvergence(
        self,
        option_string="--local-convergence",
        type=float,
        default=2,
        help="Value for convergence, default is 2",
        required=False,
        ):
        self._add_argument(dict(locals()))
    
    def add_omega(
        self,
        option_string="--omega",
        type=float,
        default=0,
        help="Weigth of the intersection in the cost function, 0 by default",
        required=False,
        ):
        self._add_argument(dict(locals()))

    def add_no_mutlistart(
        self,
        option_string="--no-multistart",
        type=bool,
        default=1,
        help="Set to 1 if you dont want to use multistart",
        required=False,
        ):
        self._add_argument(dict(locals()))
    
    def add_optimisation(
        self,
        option_string="--optimisation",
        type=str,
        default='Nelder-Mead',
        help="Optimisation method, refer to : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html ",
        required=False,
        ):
        self._add_argument(dict(locals()))
    

    #def add_threshold(
    #    self,
    #    option_string="--threshold",
    #    type=str,
    #    default=None,
    #    required=False,
    #                ):
    #    self._add_argument(dict(locals()))


    #def add_hr(
    #    self,
    #    option_string="--hr",
    #    type=str,
    #    default=None,
    #    required=False,
    #                ):
    #    self._add_argument(dict(locals()))

    #def add_hr_mask(
    #    self,
    #    option_string="--hr_mask",
    #    type=str,
    #    default=None,
    #    required=False,
    #                ):
    #    self._add_argument(dict(locals()))
        
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
