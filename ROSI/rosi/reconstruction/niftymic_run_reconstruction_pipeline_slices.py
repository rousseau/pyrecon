# -*- coding: utf-8 -*-
#Launch "run_reconstruction_pipeline_slices", a slightly different version of "run_reconstruction_pipeline" which registrer slices from the input.
#inspired from : https://github.com/gift-surg/NiftyMIC

import sys

from niftymic.application.run_reconstruction_pipeline_slices import main

if __name__ == "__main__":
    sys.exit(main())