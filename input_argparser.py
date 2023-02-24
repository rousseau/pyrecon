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
        
    def add_filenames(
        self,
        option_string="--filenames",
        nargs="+",
        help="image data in nii.gz",
        default=None,
        required=False,
                    ):
        self._add_argument(dict(locals()))
        
    def add_filenames_masks(
        self,
        option_string="--filenames_masks",
        nargs="+",
        help="mask of the data in nii.gz",
        default=None,
        required=False,
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
        option_string="--nomvt_masks",
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
        required=False,
                   ):
        self._add_argument(dict(locals()))
        
        
    def add_simulation(
        self,
        option_string="--simulation",
        nargs="+",
        help=" ",
        default=None,
        required=False,
                    ):
        self._add_argument(dict(locals()))
        
    def add_ablation(
        self,
        option_string="--ablation",
        type=str,
        default="all",
        required=False,
            ):
        self._add_argument(dict(locals()))
        
    def add_hyperparameters(
        self,
        option_string="--hyperparameters",
        nargs="+",
        default=None,
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