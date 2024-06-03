# -*- coding: utf-8 -*-
#Launch "run_reconstruction_pipeline" from NiftyMIC. I modified the script because I couldn't lunch the one from github
# src : https://github.com/gift-surg/NiftyMIC
#
import sys

from niftymic.application.run_reconstruction_pipeline import main

if __name__ == "__main__":
    sys.exit(main())